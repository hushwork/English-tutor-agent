"""会话录音 — 把一路练习会话的双向音频录成单个 16kHz 单声道 WAV。

两个写入方跑在不同线程：
- write_mic：mic 上行（16kHz PCM16，~200ms 一块，连续不断），直接追加；
- write_tts：TTS 下行（24kHz PCM16，100ms 一块，回复时突发），
  线性插值重采样到 16kHz 后追加。

设计取舍：不做精确时间对齐，按到达顺序追加。mic 流连续、间隔均匀，
本身就是天然时间轴；TTS 块插在中间，回放时间轴近似正确（TTS 突发
期间 mic 块被"推迟"到 TTS 之后，偏差在一个回复的时长量级）。换来
实现极简单、零依赖缓冲管理，对"事后回听对话"场景足够。

线程安全：内部一把 Lock 串行化两个写入线程。close() 幂等，
close 之后的写入静默忽略（stop 后残余线程再写也安全）。
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

MIC_SAMPLE_RATE = 16000   # mic 上行 / 输出文件采样率
TTS_SAMPLE_RATE = 24000   # TTS 下行采样率


class SessionRecorder:
    """单个会话的双向音频录音器（输出 16kHz 单声道 PCM16 WAV）。"""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._wf = wave.open(str(self._path), "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # 16-bit PCM
        self._wf.setframerate(MIC_SAMPLE_RATE)

    def write_mic(self, pcm: bytes) -> None:
        """追加 mic 上行音频（16kHz PCM16，无需转换）。"""
        if not pcm:
            return
        with self._lock:
            if self._closed:
                return
            self._wf.writeframes(pcm)

    def write_tts(self, pcm: bytes) -> None:
        """追加 TTS 下行音频（24kHz PCM16，线性插值重采样到 16kHz）。"""
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype=np.int16)
        if len(samples) == 0:
            return
        n_out = round(len(samples) * MIC_SAMPLE_RATE / TTS_SAMPLE_RATE)
        x_old = np.arange(len(samples))
        x_new = np.arange(n_out) * (TTS_SAMPLE_RATE / MIC_SAMPLE_RATE)
        out = np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)
        with self._lock:
            if self._closed:
                return
            self._wf.writeframes(out.tobytes())

    def close(self) -> None:
        """关闭并落盘（幂等；之后的写入静默忽略）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wf.close()
