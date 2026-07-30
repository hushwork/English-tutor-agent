# Camera Tutor 本地运行配置记录

> 日期: 2026-07-29
> 目的: 在 MacBook Pro (Intel) + Poly Sync 20 上跑通 camera-tutor 实时语音对话

---

## 环境配置

### 依赖安装

```bash
brew install portaudio                          # PyAudio 系统库
pip install pyaudio opencv-python websocket-client  # Python 依赖
pip install websockets                          # uvicorn WebSocket 支持
```

### 环境变量 (.env)

```env
DASHSCOPE_API_KEY=sk-ws-...                     # 阿里云 MaaS API Key
LLM_BASE_URL=https://llm-xo2ff9jhvnvgvu6b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OMNI_CLOUD_MODEL=qwen3.5-omni-flash-realtime    # HTTP 模式模型 (demo.py)
```

### Whisper 模型

模型 `Systran/faster-whisper-tiny` (~72MB) 下载到 HuggingFace 缓存:

```
~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny/
```

由于网络环境无法直连 huggingface.co，通过浏览器下载 zip 后手动解压到缓存目录。

---

## 代码修改

### 1. `camera_tutor/realtime_demo.py`

**ASR 语种指定** — 服务端语音识别默认偏中文，英语带口音时会被识别成中文。

- WebSocket URL 加参数: `&language=en`
- session.update 加配置: `input_audio_transcription: {language: "en"}`

**麦克风检测** — 原始代码只识别 Jabra/Brio，新增 Poly 支持 + 回退到系统默认设备。

**Whisper 离线加载** — 设置 `HF_HUB_OFFLINE=1` 避免启动时连接 HuggingFace 超时。

### 2. `camera_tutor/tutor_personas.py`

系统提示词增加语言指示: "The child speaks ENGLISH. Transcribe their speech as English."

### 3. `camera_tutor/omni_client.py`

新增 `OMNI_CLOUD_MODEL` 环境变量支持，允许通过 `.env` 指定模型名称。

### 4. `camera_tutor/dialogue.py`

修复 `CORRECTION_TEMPLATES` 中 `wrong_plural` 模板的 KeyError（模板含 `{singular}/{plural}` 但只传了 `{corrected}`）。

---

## 启动方式

```bash
cd English-tutor-agent
source venv/bin/activate

# WebSocket 实时语音对话（推荐）
python3 camera_tutor/realtime_demo.py

# 纯文本 Mock 模式（无需硬件）
python3 camera_tutor/demo.py --mock

# 家长面板
python3 camera_tutor/dashboard_server.py
# 浏览器打开 http://localhost:8200
```

---

## 已知问题

1. **WebGL 不可用** — Chrome 硬件加速被禁用，Live2D 面部页面无法渲染。需在 `chrome://settings/system` 开启"使用图形加速"。
2. **Poly Sync 20 回授** — 全双工模式下扬声器声音被麦克风重新拾取。服务端 VAD 处理，无需客户端干预。
3. **ASR 对带口音英语的识别** — 虽已指定 `language=en`，但仍可能不如母语者准确。

---

## 2026-07-30 更新：Viseme 系统优化

### 从 spectral 到 MFCC

原始的 viseme 检测使用 FFT 频谱质心 (centroid + spread) 覆盖 ~60% 准确率。
替换为 **13 维 MFCC (Mel-Frequency Cepstral Coefficients)**：

- 纯 numpy 实现 (无 scipy, 无 GPU) — ~30µs/窗
- Mel filterbank (40 通道, 预计算 + 缓存)
- DCT type-2 cepstral 系数提取

**在真实 Emma 语音 (38s, 1896 帧) 上校准**，阈值分布:
- MFCC c1 (vowel height): [-1, 31], c2 (frontness): [-11, 18]
- hf_rough (consonant detection): p50=32, p90=58
- 元音帧 67%, 辅音帧 33%, 开口率 36% (符合自然语音预期)

### 校准流程 (可选, 模型更新时)

```bash
# 1. 录制 Emma 样本
SAVE_CALIBRATION_AUDIO=1 python3 camera_tutor/realtime_demo.py
# 聊几句 → Ctrl+C → WAV 保存到 .camera-tutor-data/calibration/

# 2. 分析
python3 camera_tutor/calibrate_mfcc.py
# 输出 → 推荐阈值 → 填入 spectral_viseme.py → 重新部署
```

### 其他优化

- **ASR 漂移**: child transcript 可能不准 (尤其带口音)。仅日志保存, **不做错误检测或词汇提取**
- **词汇提取**: 从 Emma TTS 文本提取 ≥3 字母内容词, 过滤函数词和常见词
- **防重复**: 每次 Emma 回复后更新 session 指令, 注入最近 8 句话 + 高频词提醒
