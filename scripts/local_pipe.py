"""本地语音管线 — whisper（WHISPER_MODEL 可选 base/small/...）+ Kokoro TTS + 视觉"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（camera_tutor 包）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import huggingface_hub
_o = huggingface_hub.hf_hub_download
huggingface_hub.hf_hub_download = lambda *a, **kw: _o(*a, **kw, local_files_only=True)

import asyncio, base64, json, re, tempfile, threading, time, logging
from collections import deque
import numpy as np
import websockets
from faster_whisper import WhisperModel
from kokoro import KPipeline
import httpx, soundfile
from dotenv import load_dotenv
load_dotenv()

from camera_tutor.voice_gate import VoiceGate, VoiceGateConfig

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

class ConnState:
    """每连接视觉状态：帧缓冲、场景历史、最新帧——多客户端并发时互不串扰。"""
    def __init__(self):
        self.frame_buffer: deque[str] = deque(maxlen=FRAME_BUF_SIZE)
        self.scene_history: deque[str] = deque(maxlen=SCENE_HISTORY_SIZE)
        self.last_frame = {"img": None, "probed": True}

# ── 语音门禁（方案 A 文本拒识 / 方案 B KWS 唤醒，voice_gate.json 选模式） ──
# 默认 off，对现有对话零影响；dashboard 改配置后按文件 mtime 热重载，无需重启。
GATE = VoiceGate(VoiceGateConfig())          # off
_GATE_PATH = VoiceGate.default_path()
_GATE_MTIME = None

def _reload_gate_if_changed(force=False):
    global GATE, _GATE_MTIME
    try:
        mt = _GATE_PATH.stat().st_mtime
    except OSError:
        mt = None
    if not force and mt == _GATE_MTIME:
        return
    _GATE_MTIME = mt
    try:
        GATE = VoiceGate.load(_GATE_PATH)
        log.info(f"voice gate: mode={GATE.config.mode}")
    except Exception as e:
        log.warning(f"voice gate 加载失败，维持关闭: {e}")

_gate_error_logged = False

def _gate_feeds(gate, pcm_bytes) -> bool:
    """KWS 音频门：返回 False 表示未唤醒，音频应丢弃。异常时放行（不影响对话）。"""
    global _gate_error_logged
    try:
        return gate.feed_audio(np.frombuffer(pcm_bytes, dtype=np.int16))
    except Exception as e:
        if not _gate_error_logged:   # 每进程只报一次，避免刷日志
            _gate_error_logged = True
            log.warning(f"voice gate 音频门异常，持续放行: {e}")
        return True

_reload_gate_if_changed(force=True)

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

# faster-whisper 线程安全度有限，多连接经 to_thread 并发调用时串行化 transcribe
_stt_lock = threading.Lock()

# STT 拒答/解释性元话语过滤：gemma 是 chat 模型，听到非英语/噪音时爱输出
# "I'm sorry, I cannot transcribe..." 之类解释而不是空——这类文本绝不能进对话
_STT_META_RE = re.compile(
    r"cannot transcribe|can'?t transcribe|unable to transcribe|not in english|"
    r"no (clear )?(english )?speech|i'?m sorry|i apologize|"
    r"only (noise|silence)|does not contain|no audible",
    re.IGNORECASE)

def _is_stt_meta(text: str) -> bool:
    return bool(_STT_META_RE.search(text))

def stt_whisper_sync(pcm) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    soundfile.write(tmp.name, pcm, SR)
    # vad_filter=True：whisper 内置 Silero VAD 先剔除非语音区域，抑制噪音幻觉
    # 不强制 language="en"：强制英文会让 whisper 把中文语音幻听成英文。
    # 靠语言检测 + 置信度门禁甄别（whisper 的幻觉全是低置信，可挡）
    with _stt_lock:
        segs, info = get_whisper().transcribe(tmp.name, beam_size=3, vad_filter=True)
    os.unlink(tmp.name)
    if info.language != "en" or info.language_probability < 0.6:
        log.info(f"whisper 拒识: 检测到 {info.language}"
                 f"(p={info.language_probability:.2f})，非英语不入对话")
        return ""
    out = []
    for s in segs:
        # no_speech_prob 高或 avg_logprob 低的段是典型幻觉（噪音/含糊语音）
        if s.no_speech_prob > 0.6 or s.avg_logprob < -1.0:
            log.info(f"whisper 丢弃低置信段(no_speech={s.no_speech_prob:.2f} "
                     f"logprob={s.avg_logprob:.2f}): {s.text.strip()[:40]}")
            continue
        out.append(s.text.strip())
    return " ".join(out)

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
            {"type": "text", "text": "Transcribe this audio exactly. "
                                     "Output only the verbatim transcription. "
                                     "If the speech is not English, or the audio is "
                                     "only noise, music or silence, output nothing "
                                     "at all — no explanation, no apology."}]}],
        "max_tokens": 200, "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(LLM, json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
    text = clean_text(raw).strip('" ').strip()
    if text and _is_stt_meta(text):
        log.info(f"gemma 拒答文本已过滤: {text[:60]}")
        return ""
    if text and not re.search(r"[A-Za-z]{2,}", text):
        # 无实质内容（纯标点/乱码，gemma 对听不清的音频的常见输出）
        log.info(f"gemma 无效转写已过滤: {text[:40]!r}")
        return ""
    return text

async def stt(pcm) -> str:
    if STT_BACKEND == "gemma":
        try:
            return await stt_gemma(pcm)
        except Exception as e:
            log.warning(f"gemma STT 失败，回退 whisper: {e}")
    return await asyncio.to_thread(stt_whisper_sync, pcm)

if STT_BACKEND == "whisper":
    get_whisper()   # 主后端是 whisper 时启动即加载，避免首轮 STT 卡顿

kokoro_pipe = KPipeline(lang_code="a")
# kokoro_pipe 是全局单例且合成是同步密集调用——并发连接间用锁串行化，
# 避免两个协程交错驱动同一个生成器（单句 0.3-0.5s，排队可接受）
_tts_lock = asyncio.Lock()
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
    state = ConnState()   # 本连接独立的视觉状态（帧缓冲/场景历史/最新帧）
    # 本连接独立的门禁状态（KWS 开门窗口/帧缓冲/文本窗口各连接隔离）；
    # 全局 GATE 只是配置模板，热重载后（mtime 变化）重建本连接的门禁
    gate = GATE.new_session()
    gate_stamp = _GATE_MTIME
    prober = asyncio.ensure_future(scene_prober(state))

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
                    state.frame_buffer.append(img)
                    state.last_frame["img"] = img
                    state.last_frame["probed"] = False
                    # 调试落盘：每连接一个文件，避免多连接互相覆盖
                    with open(f"/tmp/camera_latest_{id(ws)}.jpg", "wb") as f:
                        f.write(base64.b64decode(img))

            elif t == "input_audio_buffer.append":
                pcm = base64.b64decode(msg.get("audio", ""))
                _reload_gate_if_changed()
                if _GATE_MTIME != gate_stamp:   # 配置热重载：重建本连接门禁
                    gate = GATE.new_session()
                    gate_stamp = _GATE_MTIME
                if not _gate_feeds(gate, pcm):
                    # 门禁关闭（未唤醒）：丢弃；半截语音清掉，防唤醒后混入旧片段
                    if buf.talking:
                        buf.get_and_clear()
                    continue
                if buf.add(pcm):
                    await process(ws, buf, instructions, voice, speed, history, state, gate)

            elif t == "input_audio_buffer.commit":
                await process(ws, buf, instructions, voice, speed, history, state, gate)

            elif t == "response.cancel":
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        prober.cancel()

async def process(ws, buf, instructions, voice=DEFAULT_VOICE, speed=0.9,
                  history=None, state=None, gate=None):
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

    # 文本门禁（方案 A）：不在指令表内则拒识——canned 话术直接 TTS（不经 LLM），否则静默
    decision = (gate or GATE).check_text(text)
    from_llm = decision.allowed
    if not decision.allowed:
        log.info(f"Gate 拒识({decision.reason}): {text}")
        if not decision.canned_reply:
            await ws.send(json.dumps({"type": "response.audio.done", "response_id": "none"}))
            return
        reply = clean_text(decision.canned_reply)
    else:
        # system prompt：原始 instructions + 场景文本历史（较早的上下文）
        sys_prompt = instructions or ""
        if state.scene_history:
            sys_prompt += ("\n\nSCENE HISTORY (what happened earlier, oldest to newest):\n"
                           + "\n".join(f"- {s}" for s in state.scene_history))

        # user content：最近 N 帧原图（旧→新）+ 文本
        images = list(state.frame_buffer)
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
            log.info(f"LLM request: images={len(images)} history={len(state.scene_history)} text={text[:30]}")
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

    # 记入对话历史（只存文本，图像不重发），供后续轮次引用；拒识话术不进历史
    if history is not None and from_llm:
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
    # 全局锁串行化：kokoro_pipe 单例的同步生成器不能被两个协程同时驱动，
    # 锁内含发送（单句 0.3-0.5s，多连接排队可接受），流式行为不变
    total = 0.0
    async with _tts_lock:
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

async def scene_prober(state):
    """周期性把最新帧发给 VLM 生成一句场景描述，滚入 scene_history。

    这样更早的画面以文本形式累积进上下文，配合 frame_buffer 里
    最近几帧原图，模拟 Qwen-Omni 会话内图像累积的效果。
    每连接一个 prober 任务，只探测本连接的 last_frame。
    """
    while True:
        await asyncio.sleep(SCENE_PROBE_INTERVAL)
        img = state.last_frame["img"]
        if not img or state.last_frame["probed"]:
            continue
        state.last_frame["probed"] = True
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
                    state.scene_history.append(f"{time.strftime('%H:%M')} {desc}")
                    log.info(f"Scene: {desc}")
        except Exception as e:
            log.warning(f"scene probe failed: {e}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_interval=None):
        log.info("Pipeline WS on :8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
