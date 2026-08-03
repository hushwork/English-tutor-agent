"""本地语音管线 — whisper-base + Kokoro TTS + 视觉"""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PIPE] %(message)s")
log = logging.getLogger("pipe")

LLM = "http://127.0.0.1:8080/v1/chat/completions"
SR = 16000
SILENCE_LIMIT = 1.0
SPEECH_THRESHOLD = 400

# 图像累积：最近几帧原图 + 场景文本历史（模拟 Qwen-Omni 的会话内图像累积）
FRAME_BUF_SIZE = 8          # 最近 N 个关键帧随语音请求一起发（旧→新）
SCENE_HISTORY_SIZE = 8      # 场景文本摘要条数上限
SCENE_PROBE_INTERVAL = 15.0 # 场景摘要探测间隔（秒）

frame_buffer: deque[str] = deque(maxlen=FRAME_BUF_SIZE)
scene_history: deque[str] = deque(maxlen=SCENE_HISTORY_SIZE)
last_frame = {"img": None, "probed": True}

whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)
kokoro_pipe = KPipeline(lang_code="a")
log.info("模型就绪")

def clean_text(text):
    # 剥掉泄漏的特殊 token：<end_of_turn>、<endofturn、<start_of_turn> 等
    text = re.sub(r'<\s*(start|end)_?of_?turn[^>]*>?', ' ', text, flags=re.IGNORECASE)
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

    def add(self, pcm_bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio**2))
        self.frames.append(audio)
        if rms > SPEECH_THRESHOLD:
            self.talking = True
            self.silence_frames = 0
        elif self.talking:
            self.silence_frames += len(audio)
        return self.talking and self.silence_frames > SR * SILENCE_LIMIT

    def get_and_clear(self):
        if not self.frames:
            return None
        audio = np.concatenate(self.frames)
        self.frames.clear()
        self.talking = False
        self.silence_frames = 0
        return (audio / 32768.0).astype(np.float32)

async def handler(ws):
    buf = VADBuffer()
    instructions = ""

    try:
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type", "")

            if t == "session.update":
                s = msg.get("session", {})
                if s.get("instructions"):
                    instructions = s["instructions"]
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
                    await process(ws, buf, instructions)

            elif t == "input_audio_buffer.commit":
                await process(ws, buf, instructions)

            elif t == "response.cancel":
                pass
    except websockets.ConnectionClosed:
        pass

async def process(ws, buf, instructions):
    pcm = buf.get_and_clear()
    if pcm is None or len(pcm) < SR * 0.3:
        return

    t0 = time.time()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    soundfile.write(tmp.name, pcm, SR)
    segs, _ = whisper.transcribe(tmp.name, language="en", beam_size=3, vad_filter=False)
    os.unlink(tmp.name)
    text = " ".join(s.text.strip() for s in segs)
    if not text:
        return
    log.info(f"STT({time.time()-t0:.1f}s): {text}")
    await ws.send(json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": text}))

    # system prompt：原始 instructions + 场景文本历史（较早的上下文）
    sys_prompt = instructions or ""
    if scene_history:
        sys_prompt += ("\n\nSCENE HISTORY (what happened earlier, oldest to newest):\n"
                       + "\n".join(f"- {s}" for s in scene_history))

    # user content：最近 N 帧原图（旧→新）+ 文本
    images = list(frame_buffer)
    async with httpx.AsyncClient(timeout=60) as cli:
        if images:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in images
            ]
            content.append({"type": "text", "text": (
                f"These are the last {len(images)} camera frames (oldest to newest). "
                f"The child said: {text}. Respond naturally.")})
            payload = {
                "model": "gemma4",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content},
                ], "max_tokens": 40,
            }
        else:
            payload = {
                "model": "gemma4",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text},
                ], "max_tokens": 40,
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

    await ws.send(json.dumps({"type": "response.audio_transcript.delta", "delta": reply}))
    await ws.send(json.dumps({"type": "response.audio_transcript.done", "transcript": reply}))

    chunks = []
    for _gs, _ps, audio in kokoro_pipe(reply, voice="af_heart", speed=0.9):
        chunks.append(audio)
    tts_audio = np.concatenate(chunks)
    tts_audio = (tts_audio * 32768).astype(np.int16)

    rid = f"r_{int(time.time()*1000)}"
    await ws.send(json.dumps({"type": "response.created", "response": {"id": rid}}))
    for i in range(0, len(tts_audio), 2400):
        await ws.send(json.dumps({
            "type": "response.audio.delta",
            "response_id": rid,
            "delta": base64.b64encode(tts_audio[i:i+2400].tobytes()).decode(),
        }))
    await ws.send(json.dumps({"type": "response.audio.done", "response_id": rid}))
    log.info(f"TTS({time.time()-t0:.1f}s): {len(tts_audio)/24000:.1f}s")

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
