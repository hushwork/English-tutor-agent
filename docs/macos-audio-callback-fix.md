# macOS Audio Fix — Callback-based Streams

## 问题

在 macOS 上运行 `camera_tutor/realtime_demo.py` 时，Qwen-Omni LLM 无法听清用户的语音输入，对话质量极差。但同样代码在 Ubuntu 上一切正常。

## 排查过程

### 第一层：误判为增益不足

日志显示 mic level RMS 7-24（int16 量程，max=32767）。正常语音应在此量程上达到 RMS 500-3000。初步认为 macOS CoreAudio 输入增益太低，将 `DEFAULT_MIC_GAIN` 从 1x 调到 2x 再调到 10x。

**诊断脚本** (`camera_tutor/audio_diagnostic.py`) 实测结果推翻了这一假设：

```
Noise floor:              RMS=25
Speech (overall):         RMS=1620, Peak=17589
SNR:                      40.4 dB
1x gain:                  RMS=1620, Peak=17589  ✅ Loud & clear
2x gain:                  Peak hits 32767       ⚠️ CLIPPING!
```

Jabra Evolve2 65 在 macOS 上原始信号就已经很好。`RMS 7-24` 是 5 秒滑动平均里包含了大量静音段导致的假象。增益调高只会削波失真。

→ **回退 `DEFAULT_MIC_GAIN` 到 1.0x。**

### 第二层：发现真正的元凶 — Buffer Overflow

用户在诊断脚本中反馈"录音断断续续"。加入 dropout 检测后：

```
Chunks recorded: 14 (expected 40)    ← 8 秒丢了 65% 音频
Overflows:       14 / 14             ← 每个 chunk 都溢出
Avg read time:   613 ms (expected 200 ms)  ← 读操作慢 3 倍
```

**根因**：macOS CoreAudio 上，`sounddevice.RawInputStream` 在小 blocksize（3200 frames = 200ms @ 16kHz）下，`read()` 阻塞时间远超预期（613ms vs 200ms）。PortAudio 内部缓冲区在此期间溢出，大量音频数据被丢弃。

这就是 LLM 听不清的真正原因 —— 收到的音频是碎片化的，中间有大段空白。

Linux ALSA 不受此影响，所以 Ubuntu 上正常。

→ **mic 端：`RawInputStream` → callback-based `InputStream`**

### 第三层：播放端同样的问题

录音修复后 ASR 准确率大幅提升，但 Emma（TTS）的声音开始断断续续。

镜像问题：**`RawOutputStream.write()` 同步阻塞**。每次 write 卡住主线程 100-200ms，期间无法处理 WebSocket 消息，下一个音频 delta chunk 延迟到达 → 播放有 gap。

→ **spk 端：`RawOutputStream` → callback-based `OutputStream`**

### 附带修复

| 问题 | 修复 |
|------|------|
| `Error append image before append audio` | mic + vision 等 `session.updated` 后才开始发送 |
| Emma 连续重复同一句话（vision-only loop） | `_build_instructions` 检测连续相同回复，注入 CRITICAL 指令 |
| 服务端 VAD 阈值偏高 | 从 0.7 降到 0.5 |

## 架构变化

```
之前 (RawStream — 阻塞式):
  RawInputStream.read()   → 阻塞 613ms（macOS bug）
  RawOutputStream.write() → 阻塞主线程，卡住 WebSocket

之后 (Callback — 事件驱动):
  InputStream(callback)   → PortAudio 50ms 回调 → ring buffer → read_mic()
  OutputStream(callback)  → write_spk() 推入 ring buffer → PortAudio 回调消费
```

**为什么 callback 模式在所有平台都更好：**

1. **解耦 I/O 和音频时钟** — 回调按硬件时钟触发，不受主线程繁忙影响
2. **天然抗 jitter** — 网络波动不影响播放连续性（ring buffer 做缓冲）
3. **跨平台一致** — Linux/Windows/macOS 行为完全一致

代价：多一个后台线程 + deque，CPU 开销可忽略。

## 新增文件

- `camera_tutor/audio_diagnostic.py` — 麦克风诊断工具
  - 分步录制环境噪音 + 语音
  - 实时电平表
  - SNR 信噪比计算
  - 增益模拟（找到最佳增益值）
  - callback 间隔检测（发现掉帧）
  - WAV 文件导出供人工检查

## 使用诊断工具

```bash
source venv/bin/activate
python3 camera_tutor/audio_diagnostic.py          # 默认麦克风
python3 camera_tutor/audio_diagnostic.py 3        # 指定设备 index
```

## 配置参数

可通过 `.env` 覆盖：

```
MIC_GAIN=1.0        # 麦克风增益（线性，默认 1.0）
```
