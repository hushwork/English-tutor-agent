"""本地语音管线 — whisper（WHISPER_MODEL 可选 base/small/...）+ Kokoro TTS + 视觉"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import huggingface_hub
_o = huggingface_hub.hf_hub_download
huggingface_hub.hf_hub_download = lambda *a, **kw: _o(*a, **kw, local_files_only=True)

import asyncio, base64, json, re, tempfile, time, logging
from collections import deque
import numpy as np
import websockets
from faster_whisper import WhisperModel
from kokoro import KPipeline
import httpx, soundfile
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PIPE] %(message)s")
log = logging.getLogger("pipe")

LLM = "http://127.0.0.1:8080/v1/chat/completions"
SR = 16000
# 说完话后的静音判定时长（秒），VAD_SILENCE 可调；太短会把长停顿切成两段
SILENCE_LIMIT = float(os.environ.get("VAD_SILENCE", "0.7"))
# RMS 语音检测阈值（VAD_THRESHOLD 可调）。注意：AGC 会把噪音floor放大到
# ~2600，与本阈值冲突——使用本管道时应关闭麦克风 AGC。
# 安静房间（噪音floor ~50）下 400 是经验值；噪音大环境再上调。
SPEECH_THRESHOLD = int(os.environ.get("VAD_THRESHOLD", "400"))

DEFAULT_VOICE = "af_heart"
# 人格音色（Qwen-Omni 云端音色名）→ 本地 Kokoro 近似音色
KOKORO_VOICE_MAP = {
    "mia": "af_heart",      # Emma — 温暖陪伴
    "serena": "af_sarah",   # Serena — 温柔安静
    "momo": "af_sky",       # Bella — 活泼搞怪
    "tina": "af_bella",     # Sophie — 甜甜暖暖
    "mione": "af_nicole",   # Olivia — 知性（本地无英伦音，取近似）
    "jennifer": "af_jessica",  # Grace — 面试教练（专业知性）
}

def map_voice(name):
    """Map a persona/cloud voice name to a local Kokoro voice."""
    if not name:
        return DEFAULT_VOICE
    mapped = KOKORO_VOICE_MAP.get(name.lower())
    if mapped:
        return mapped
    # 已是 Kokoro 音色名（形如 af_xxx）则透传，否则回退默认
    return name if "_" in name else DEFAULT_VOICE

# 图像累积：最近几帧原图 + 场景文本历史（模拟 Qwen-Omni 的会话内图像累积）
FRAME_BUF_SIZE = 2          # 最近 N 个关键帧随语音请求一起发（旧→新）；多帧极占上下文，2 帧够用
SCENE_HISTORY_SIZE = 4      # 场景文本摘要条数上限
SCENE_PROBE_INTERVAL = 15.0 # 场景摘要探测间隔（秒）
HISTORY_TURNS = 4           # 随请求携带的最近对话轮数（1 轮 = user + assistant 各一条）

frame_buffer: deque[str] = deque(maxlen=FRAME_BUF_SIZE)
scene_history: deque[str] = deque(maxlen=SCENE_HISTORY_SIZE)
last_frame = {"img": None, "probed": True}

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # base/small/medium
# STT 后端：gemma = 音频直送本地 LLM 音频输入（实测对儿童语音最准，延迟略高）；
#           whisper = faster-whisper 本地转写（快，小模型误差大）
STT_BACKEND = os.environ.get("STT_BACKEND", "gemma")

_whisper = None
def get_whisper():
    """whisper 懒加载：STT_BACKEND=gemma 时只在兜底时才加载，不占启动时间和显存。"""
    global _whisper
    if _whisper is None:
        try:
            _whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            log.info(f"whisper[{WHISPER_MODEL}]: CUDA")
        except Exception as e:
            log.warning(f"whisper CUDA 不可用，回退 CPU: {e}")
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=2)
    return _whisper

def stt_whisper(pcm) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    soundfile.write(tmp.name, pcm, SR)
    # vad_filter=True：whisper 内置 Silero VAD 先剔除非语音区域，抑制噪音幻觉
    segs, _ = get_whisper().transcribe(tmp.name, language="en", beam_size=3, vad_filter=True)
    os.unlink(tmp.name)
    return " ".join(s.text.strip() for s in segs)

async def stt_gemma(pcm) -> str:
    """整段语音送本地 LLM 的音频输入转写（llama-server 多模态）。"""
    import io as _io
    buf = _io.BytesIO()
    soundfile.write(buf, pcm, SR, format="WAV")
    payload = {
        "model": "gemma4",
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}},
            {"type": "text", "text": "Transcribe this English audio exactly. "
                                     "Output only the transcription. "
                                     "If the audio contains no clear English speech "
                                     "(only noise or silence), output nothing."}]}],
        "max_tokens": 200, "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(LLM, json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
    return clean_text(raw).strip('" ').strip()

async def stt(pcm) -> str:
    if STT_BACKEND == "gemma":
        try:
            return await stt_gemma(pcm)
        except Exception as e:
            log.warning(f"gemma STT 失败，回退 whisper: {e}")
    return stt_whisper(pcm)

if STT_BACKEND == "whisper":
    get_whisper()   # 主后端是 whisper 时启动即加载，避免首轮 STT 卡顿

kokoro_pipe = KPipeline(lang_code="a")
try:
    import torch
    if torch.cuda.is_available():
        kokoro_pipe.model = kokoro_pipe.model.to("cuda")
        log.info("kokoro: CUDA")
except Exception as e:
    log.warning(f"kokoro CUDA 不可用，回退 CPU: {e}")
log.info("模型就绪")

def clean_text(text):
    # 剥掉泄漏的特殊 token：<end_of_turn>、</startofturn、<start_of_turn> 等
    text = re.sub(r'<\s*/?\s*(start|end)_?of_?turn[^>]*>?', ' ', text, flags=re.IGNORECASE)
    # 剥掉舞台指令/状态描述：(smiles)、(laughing)、（微笑）等，TTS 不应念出来
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'（[^）]*）', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\*.*?\*', '', text)
    text = re.sub(r'[#*~_>`]', '', text)
    text = text.replace('\n', ' ').strip()
    return ' '.join(text.split())

class VADBuffer:
    def __init__(self):
        self.frames = []
        self.talking = False
        self.silence_frames = 0
        self.speech_frames = 0   # 累计超过阈值的样本数（区分真语音和瞬时尖峰）
        # 说话前的环境音只保留 3 块（0.6s）作 pre-roll——否则长时间噪音会
        # 混进送 STT 的段里，Gemma 音频转写会对着噪音编造内容
        self._preroll: deque = deque(maxlen=3)
        self._dbg_max = 0.0
        self._dbg_n = 0

    def add(self, pcm_bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio**2))
        # 电平遥测：每 ~5s（25 个 200ms 块）打一次峰值，便于调 VAD 阈值
        self._dbg_max = max(self._dbg_max, rms)
        self._dbg_n += 1
        if self._dbg_n >= 25:
            log.info(f"mic level: max_rms={self._dbg_max:.0f} "
                     f"(threshold={SPEECH_THRESHOLD}, talking={self.talking})")
            self._dbg_max = 0.0
            self._dbg_n = 0
        if rms > SPEECH_THRESHOLD:
            if not self.talking:
                # 段开始：带上 pre-roll 作上下文
                self.frames = list(self._preroll)
                self._preroll.clear()
                self.speech_frames = 0
            self.talking = True
            self.silence_frames = 0
            self.speech_frames += len(audio)
            self.frames.append(audio)
        elif self.talking:
            self.silence_frames += len(audio)
            self.frames.append(audio)
        else:
            self._preroll.append(audio)
        return self.talking and self.silence_frames > SR * SILENCE_LIMIT

    def get_and_clear(self):
        if not self.frames:
            return None, 0
        audio = np.concatenate(self.frames)
        speech = self.speech_frames
        self.frames.clear()
        self.talking = False
        self.silence_frames = 0
        self.speech_frames = 0
        return (audio / 32768.0).astype(np.float32), speech

async def handler(ws):
    buf = VADBuffer()
    instructions = ""
    voice = DEFAULT_VOICE
    speed = 0.9   # TTS 语速，可被 session.update 按角色覆盖
    # 对话历史（纯文本）：没有它模型每轮都是"失忆"状态，答非所问、反复重开话题
    history: deque = deque(maxlen=HISTORY_TURNS * 2)

    try:
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type", "")

            if t == "session.update":
                s = msg.get("session", {})
                if s.get("instructions"):
                    instructions = s["instructions"]
                out = (s.get("audio") or {}).get("output", {})
                v = out.get("voice", "")
                if v:
                    nv = map_voice(v)
                    if nv != voice:
                        log.info(f"Voice: {voice} → {nv} (persona voice: {v})")
                        voice = nv
                sp = out.get("speed")
                if sp:
                    ns = max(0.5, min(2.0, float(sp)))
                    if ns != speed:
                        log.info(f"Speed: {speed} → {ns} (persona 语速)")
                        speed = ns
                await ws.send(json.dumps({"type": "session.created", "session": {"type": "realtime"}}))

            elif t == "input_image_buffer.append":
                img = msg.get("image")
                if img:
                    frame_buffer.append(img)
                    last_frame["img"] = img
                    last_frame["probed"] = False
                    with open("/tmp/camera_latest.jpg", "wb") as f:
                        f.write(base64.b64decode(img))

            elif t == "input_audio_buffer.append":
                pcm = base64.b64decode(msg.get("audio", ""))
                if buf.add(pcm):
                    await process(ws, buf, instructions, voice, speed, history)

            elif t == "input_audio_buffer.commit":
                await process(ws, buf, instructions, voice, speed, history)

            elif t == "response.cancel":
                pass
    except websockets.ConnectionClosed:
        pass

async def process(ws, buf, instructions, voice=DEFAULT_VOICE, speed=0.9,
                  history=None):
    pcm, speech = buf.get_and_clear()
    # 双保险：整段至少 0.3s，且其中"超过阈值"的音频至少 0.4s——
    # 单个噪音尖峰（风扇/电流声）凑不够 0.4s，直接丢弃，避免 whisper 对着纯噪音幻觉出句子
    if pcm is None or len(pcm) < SR * 0.3 or speech < SR * 0.4:
        if pcm is not None and speech < SR * 0.4:
            log.info(f"VAD drop: speech={speech/SR:.2f}s < 0.4s（噪音尖峰，不送 STT）")
        return

    t0 = time.time()
    text = await stt(pcm)
    if not text:
        return
    log.info(f"STT[{STT_BACKEND}]({time.time()-t0:.1f}s): {text}")
    await ws.send(json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": text}))

    # system prompt：原始 instructions + 场景文本历史（较早的上下文）
    sys_prompt = instructions or ""
    if scene_history:
        sys_prompt += ("\n\nSCENE HISTORY (what happened earlier, oldest to newest):\n"
                       + "\n".join(f"- {s}" for s in scene_history))

    # user content：最近 N 帧原图（旧→新）+ 文本
    images = list(frame_buffer)
    # messages：system + 最近几轮对话历史（纯文本，不带图）+ 当前输入
    msgs = [{"role": "system", "content": sys_prompt}]
    if history:
        msgs.extend(history)
    async with httpx.AsyncClient(timeout=60) as cli:
        if images:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in images
            ]
            content.append({"type": "text", "text": (
                f"These are the last {len(images)} camera frames (oldest to newest). "
                f"The child said: {text}. Respond naturally.")})
            msgs.append({"role": "user", "content": content})
            payload = {
                "model": "gemma4",
                "messages": msgs,
            }
        else:
            msgs.append({"role": "user", "content": text})
            payload = {
                "model": "gemma4",
                "messages": msgs,
            }
        log.info(f"LLM request: images={len(images)} history={len(scene_history)} text={text[:30]}")
        resp = await cli.post(LLM, json=payload)
        if resp.status_code != 200:
            reply = "Sorry, I did not catch that."
        else:
            try:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                reply = "Sorry, I missed that."
    reply = clean_text(reply)
    log.info(f"LLM({time.time()-t0:.1f}s): {reply}")

    if not reply:
        # 整段回复都是舞台指令，剥完为空——跳过本次播报
        await ws.send(json.dumps({"type": "response.audio.done", "response_id": "none"}))
        return

    # 记入对话历史（只存文本，图像不重发），供后续轮次引用
    if history is not None:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})

    await ws.send(json.dumps({"type": "response.audio_transcript.delta", "delta": reply}))
    await ws.send(json.dumps({"type": "response.audio_transcript.done", "transcript": reply}))

    rid = f"r_{int(time.time()*1000)}"
    await ws.send(json.dumps({"type": "response.created", "response": {"id": rid}}))

    async def send_audio(pcm_f32):
        """Slice one synthesized chunk into 100ms deltas and stream it."""
        if hasattr(pcm_f32, "detach"):  # kokoro CUDA 模式产出的是 torch Tensor
            pcm_f32 = pcm_f32.detach().float().cpu().numpy()
        pcm16 = (np.asarray(pcm_f32, dtype=np.float32) * 32768).astype(np.int16)
        for i in range(0, len(pcm16), 2400):
            await ws.send(json.dumps({
                "type": "response.audio.delta",
                "response_id": rid,
                "delta": base64.b64encode(pcm16[i:i+2400].tobytes()).decode(),
            }))

    # 边合成边发：kokoro 逐句产出，首句合成完就开始出声
    total = 0.0
    try:
        for _gs, _ps, audio in kokoro_pipe(reply, voice=voice, speed=speed):
            total += len(audio) / 24000
            await send_audio(audio)
    except Exception as e:
        if total == 0:
            # 音色不可用（如离线缓存缺失）时回退默认音色，不让对话中断
            log.warning(f"TTS voice {voice} failed ({e}) — fallback to {DEFAULT_VOICE}")
            for _gs, _ps, audio in kokoro_pipe(reply, voice=DEFAULT_VOICE, speed=speed):
                total += len(audio) / 24000
                await send_audio(audio)
        else:
            log.warning(f"TTS mid-stream error: {e}")
    await ws.send(json.dumps({"type": "response.audio.done", "response_id": rid}))
    log.info(f"TTS({time.time()-t0:.1f}s): {total:.1f}s")

async def scene_prober():
    """周期性把最新帧发给 VLM 生成一句场景描述，滚入 scene_history。

    这样更早的画面以文本形式累积进上下文，配合 frame_buffer 里
    最近几帧原图，模拟 Qwen-Omni 会话内图像累积的效果。
    """
    while True:
        await asyncio.sleep(SCENE_PROBE_INTERVAL)
        img = last_frame["img"]
        if not img or last_frame["probed"]:
            continue
        last_frame["probed"] = True
        try:
            async with httpx.AsyncClient(timeout=60) as cli:
                resp = await cli.post(LLM, json={
                    "model": "gemma4",
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                        {"type": "text", "text": (
                            "In ONE short sentence: what is the child doing, "
                            "and what objects do you see?")},
                    ]}],
                    "max_tokens": 40,
                })
                if resp.status_code != 200:
                    log.warning(f"scene probe HTTP {resp.status_code}")
                    continue
                desc = clean_text(resp.json()["choices"][0]["message"]["content"])
                if desc:
                    scene_history.append(f"{time.strftime('%H:%M')} {desc}")
                    log.info(f"Scene: {desc}")
        except Exception as e:
            log.warning(f"scene probe failed: {e}")

async def main():
    asyncio.ensure_future(scene_prober())
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_interval=None):
        log.info("Pipeline WS on :8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
