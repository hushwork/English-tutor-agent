"""麦克风录音测试 — 保存WAV检查录音质量"""
import sounddevice as sd, numpy as np, wave, sys, os

DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SR = 16000
DEV = 8  # Poly Sync 20-M

print(f"🎤 Poly Sync 20 录音 {DUR}秒...")
audio = sd.rec(int(SR*DUR), samplerate=SR, device=DEV, channels=1, dtype='int16')
sd.wait()

rms = np.sqrt(np.mean(audio.astype(float)**2))
peak = np.max(np.abs(audio))
print(f"RMS: {rms:.0f}  Peak: {peak}  (0=静音)")

path = os.path.join(os.path.dirname(__file__), "..", ".camera-tutor-data", "calibration", "mic_test.wav")
os.makedirs(os.path.dirname(path), exist_ok=True)
with wave.open(path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes(audio.tobytes())
print(f"✅ 保存到: {path}")
