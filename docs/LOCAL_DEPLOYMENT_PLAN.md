# 本地化部署完成状态

> 更新日期：2026-08-05

## 当前架构

```
realtime_demo.py ──ws──▶ local_pipe.py (:8765) ──▶ llama-server (:8080)
                            │
              VAD → whisper STT (CUDA) → LLM (GPU) → Kokoro TTS (CUDA)
```

默认配置端到端对话延迟约 3 秒（VAD 0.7s + STT ~0.3s + LLM ~1.7s + TTS 流式出声）。

## 支持的模型

| 模型 | 启动命令 | GPU | 速度 | 视觉 |
|------|---------|-----|------|------|
| Gemma 4 E4B | `bash scripts/start-llm.sh e4b` | ~4.4GB | 75 tok/s | ✅ 需 mmproj |
| Gemma 4 E2B | `bash scripts/start-llm.sh e2b` | ~1.7GB | 133 tok/s | ⚠️ 未测 |
| Qwen3-VL 8B | `bash scripts/start-llm.sh qwen` | ~6.3GB | 51 tok/s | ✅ 完整 |

## 切换模型

```bash
bash scripts/switch-model.sh e4b  # Gemma E4B（默认）
bash scripts/switch-model.sh qwen  # Qwen3-VL
```

## 启动步骤

一键启停（推荐）：

```bash
bash scripts/start-all.sh start    # 拉起 llama-server + dashboard + local_pipe + realtime_demo
bash scripts/start-all.sh status   # 查看四个组件状态
bash scripts/start-all.sh stop     # 全部停止
```

手动分步启动（调试用）：

```bash
# 1. 启动 LLM
bash scripts/start-llm.sh e4b

# 2. 启动语音管线
nohup .venv/bin/python3 scripts/local_pipe.py > logs/local_pipe.log 2>&1 &

# 3. 运行（设备选择会保存到 .camera-tutor-data/devices.json，下次自动复用）
.venv/bin/python camera_tutor/realtime_demo.py --select-devices
```

## 日志说明

| 日志文件 | 进程 | 看什么 |
|---------|------|--------|
| `logs/realtime_demo.log` | agent 主程序 | 设备选择、连接状态、`👧 Child:` / `🤖 Serena:` 对话结果、新词汇记录、麦克风异常告警 |
| `logs/local_pipe.log` | 语音管线 | 每轮耗时分解 `STT/LLM/TTS(x.xs)`、`LLM request: images=N`、`Scene:` 场景描述、`mic level` 电平遥测、音色切换 |
| `logs/llama.log` | llama-server | 模型加载、推理服务状态 |
| `logs/dashboard.log` | 家长面板 | HTTP 服务 |

一句话：**看"发生了什么对话"用 realtime_demo.log；看"快不快、卡在哪段"用 local_pipe.log**。

```bash
tail -f logs/local_pipe.log      # 调延迟/识别问题时最常用
```

## 调参（环境变量，写入 .env 即可）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VAD_THRESHOLD` | 400 | RMS 语音检测阈值。安静房间 400 合适；噪音大再上调 |
| `VAD_SILENCE` | 0.7 | 说完话后的静音判定时长（秒）。太短会把长停顿切成两段 |
| `MIC_GAIN` | 1.0 | 麦克风数字增益，由 audio_diagnostic 推荐 |
| `AV_SOURCE` | local | `webrtc` = 远程浏览器采集音视频（见 README WebRTC 章节） |
| `VISME_LEAD_MS` | 80 | WebRTC 模式唇形同步补偿（毫秒） |

⚠️ **使用本地管线时必须关闭麦克风 AGC**（设备选择时不要开 `--agc`）：AGC 会把噪音底放大到 ~2600 RMS，导致 RMS VAD 永远处于"说话中"，永不切段、永不回复。

## 音频诊断

```bash
.venv/bin/python camera_tutor/audio_diagnostic.py [设备号]   # 四步交互：噪音底→8s 说话→两次回放，输出 VERDICT 和推荐 MIC_GAIN
.venv/bin/python scripts/mic_test.py                          # 快速麦克风测试
.venv/bin/python camera_tutor/calibrate_mfcc.py               # 唇形 MFCC 校准分析
```

诊断前建议先退出主程序释放设备：`pkill -f 'camera_tutor/realtime_demo.py'`

## 常见排障

**说话后完全没有回复（日志无 STT 行）**
1. 看 `logs/local_pipe.log` 的 `mic level: max_rms=` 遥测：
   - RMS 1~8（-90dBFS）= 无信号：检查耳机电源、麦克风杆是否放下（Jabra 翻起即硬件静音）、PipeWire 音量（`wpctl status`）
   - RMS 持续 >400 且 talking=True = AGC 开着或噪音太大，见上面的 AGC 警告
   - 说话时有峰值但 < VAD_THRESHOLD = 阈值过高，`VAD_THRESHOLD` 调低
2. 设备编号重插后会漂移，保存的配置按设备名自动恢复；选错设备用 `--select-devices` 重选

**llama-server 启动即崩（Vulkan ErrorOutOfDeviceMemory）**
本机 Vulkan 偶发上报损坏的显存值（16EB 假值）。直接重跑一次通常就好；持续崩溃则把 `start-all.sh` 里的 `--n-gpu-layers all` 临时改为 `--device none`（CPU 推理，延迟约 60s/轮）。

**有 STT/LLM 日志但没有声音**
检查 TTS 行是否报错（音色缺失会自动回退 af_heart）；确认扬声器输出设备选的是 Jabra/耳机而不是 HDMI。

## 已知问题

1. **Gemma E4B 视觉**：不支持跨轮视觉记忆，每轮独立看图
2. **Gemma E4B 旁白**：偶尔生成 `(描述动作)` 括号内容
3. **STT 精度**：whisper-base 对儿童口音识别率有限
4. **模型切换**：换模型需重启 llama-server 和 pipe
5. **实时打断**：本地管线不支持 barge-in，需等回复播完

## 回退云端

注释掉 `.env` 里的 `OMNI_WS_URL` 并确保 `DASHSCOPE_API_KEY` 有效，重启 realtime_demo 即可（语音走 Qwen-Omni 云端实时接口，延迟更低、识别更准）。
