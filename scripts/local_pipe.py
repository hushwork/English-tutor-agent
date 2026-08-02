"""本地语音管线 — whisper-base + Kokoro TTS + 视觉"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import huggingface_hub
_o = huggingface_hub.hf_hub_download
huggingface_hub.hf_hub_download = lambda *a, **kw: _o(*a, **kw, local_files_only=True)

import asyncio, base64, json, re, tempfile, time, logging
import numpy as np
import websockets
from faster_whisper import WhisperModel
from kokoro import KPipeline
import httpx, soundfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PIPE] %(message)s")
log = logging.getLogger("pipe")

LLM = "http://127.0.0.1:8080/v1/chat/completions"
SR = 16000
SILENCE_LIMIT = 0.6
SPEECH_THRESHOLD = 400

whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)
kokoro_pipe = KPipeline(lang_code="a")
log.info("模型就绪")

def clean_text(text):
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
    latest_image = None

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
                latest_image = msg.get("image")
                with open("/tmp/camera_latest.jpg", "wb") as f:
                    f.write(base64.b64decode(latest_image))

            elif t == "input_audio_buffer.append":
                pcm = base64.b64decode(msg.get("audio", ""))
                if buf.add(pcm):
                    await process(ws, buf, instructions, latest_image)

            elif t == "input_audio_buffer.commit":
                await process(ws, buf, instructions, latest_image)

            elif t == "response.cancel":
                pass
    except websockets.ConnectionClosed:
        pass

async def process(ws, buf, instructions, image):
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

    async with httpx.AsyncClient(timeout=30) as cli:
        if image:
            payload = {
                "model": "gemma4",
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                        {"type": "text", "text": f"The child said: {text}. Respond naturally."}
                    ]},
                ], "max_tokens": 40,
            }
        else:
            payload = {
                "model": "gemma4",
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": text},
                ], "max_tokens": 40,
            }
        log.info(f"LLM request: image={bool(image)} text={text[:30]}")
        if image:
            h = __import__("hashlib").md5(image.encode()).hexdigest()[:8]
            with open(f"/tmp/frame_{h}.jpg", "wb") as f:
                f.write(base64.b64decode(image))
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

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_interval=None):
        log.info("Pipeline WS on :8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
