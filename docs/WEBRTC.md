# WebRTC 远程设备模式

> 用一台远程浏览器（手机 / 平板 / 另一台电脑）替代本地麦克风、扬声器、摄像头，
> agent 端的语音管线（VAD / ASR / LLM / TTS / 视觉）完全无感知。

引入提交：`7cee578`（2026-08-04）。核心实现：`camera_tutor/rtc_device.py`（536 行）。

---

## 1. 设计思路：接缝替身（Seam Classes）

WebRTC 模式不改任何对话逻辑，而是用三个 duck-typed 类替换本地硬件访问，
接口与本地实现逐一对应：

| 替身类 | 替换对象 | 接口 |
|--------|---------|------|
| `RTCAudioManager` | `audio_manager.AudioManager` | `read_mic()` → 16kHz PCM16（~200ms）<br>`write_spk(24kHz PCM16, visemes)` |
| `RTCFrameSource` | `camera.CameraPipeline` | `read_frame()` → `(ok, BGR ndarray)` |
| `RTCDeviceManager` | —（新增） | 持有 peer connection 与上面两个替身 |

接线点在 `camera_tutor/agent.py` `setup()`：`av_source="webrtc"` 时
`self.audio` / `self.camera` 直接指向 RTC 替身，之后的 VAD/ASR/LLM/TTS/视觉
代码完全复用。

### 线程模型

- `RTCPeerConnection` 和所有 track I/O 都活在 **uvicorn 事件循环**里
  （`/rtc/offer` 端点在该循环中运行）。
- agent 的工作线程只接触带锁的 `deque`/`bytearray`，**没有跨事件循环调用**。

## 2. 信令

- 方式：**HTTP 一次性 offer/answer**（非 WebSocket）。
  浏览器 POST SDP offer 到 `POST /rtc/offer`（`dashboard_server.py:375`），同步返回 answer。
- 前端不 Trickle：先等 ICE gathering 完成再发 offer（`face_preview.html:98`）。
- 无 STUN/TURN：**仅限局域网**（`face_preview.html:81`）。
- 可选鉴权：环境变量 `RTC_TOKEN` 设置后，`/rtc/offer` 要求 `Authorization: Bearer <token>`；
  页面侧用 `?token=xxx` 传递。
- 单 peer：新 offer 会顶掉旧连接（`rtc_device.py:481-483`）。
- **dashboard 必须与 agent 同进程**：RTC manager 通过模块级注册表
  （`set_rtc_manager`）在同进程内共享。若端口已被独立 dashboard 占用，
  WebRTC 模式下 agent 会报错退出（`agent.py:636-643`）。

## 3. 媒体通路

### 麦克风上行（浏览器 → agent）

```
浏览器 getUserMedia(48kHz) → aiortc 音频轨
  → _consume_mic 重采样 48k→16k（av.AudioResampler）
  → 线程安全缓冲（上限 ~5s，溢出丢最旧）
  → read_mic() 返回 200ms PCM16（应用 MIC_GAIN，跟踪 RMS 电平）
```

### 扬声器下行（agent → 浏览器）

```
TTS 24kHz PCM + visemes → write_spk() → 环形缓冲（200 块）
  → _TTSOutTrack.recv() 由 aiortc 按 20ms 实时节拍驱动
  → 每 tick 取 480 采样，确定性 2x 线性上采样到 960 @ 48kHz
  → 欠载补零（保持 WebRTC 时钟稳定）；无 peer 时直接丢弃
```

> 为什么不用 `av.AudioResampler` 做出方向上采样：resampler 有滤波延迟，
> 不能保证每次调用严格对齐 20ms 边界，因此用确定性线性插值。

### 摄像头上行（浏览器 → agent）

```
浏览器视频轨 → _consume 协程持续 recv（排空解码器）
  → to_ndarray 节流 0.2s，只保留最新 BGR 帧
  → read_frame() 返回最新帧的拷贝
```

### Viseme 唇形同步

不走 RTCDataChannel。viseme 与音频块在 `write_spk()` 时配对，在音频交给
aiortc 发送的时刻 + `VISME_LEAD_MS`（默认 80ms，补偿浏览器播放缓冲）由独立
drain 线程触发，经既有 WebSocket `/ws/emma/face` 推给 Live2D 页面。

## 4. 使用方法

```bash
# 1. 配置（.env）
AV_SOURCE=webrtc
# 远程浏览器（非 localhost）getUserMedia 要求安全上下文 → 配置 TLS（mkcert 签发）
DASHBOARD_TLS_CERT=/path/to/cert.pem
DASHBOARD_TLS_KEY=/path/to/key.pem
# 可选：信令鉴权
RTC_TOKEN=some-secret

# 2. 启动 agent（dashboard 随 agent 同进程拉起）
.venv/bin/python camera_tutor/realtime_demo.py --av-source webrtc

# 3. 远程浏览器打开设备页面，点击开始
#    https://<agent-ip>:8200/static/face_preview.html?device=1&token=some-secret
```

页面行为：请求麦克风+摄像头权限 → 本地回显画面 → 建立连接；
断开（failed/disconnected）后 2 秒自动重连。

### 配置项汇总

| 变量 | 默认 | 说明 |
|------|------|------|
| `AV_SOURCE` | `local` | `webrtc` 启用远程设备模式 |
| `VISME_LEAD_MS` | 80 | 唇形同步补偿（毫秒） |
| `RTC_TOKEN` | 空（不鉴权） | `/rtc/offer` 的 Bearer 令牌 |
| `DASHBOARD_TLS_CERT` / `DASHBOARD_TLS_KEY` | 空 | HTTPS 证书（远程浏览器必需） |

## 5. 测试

三个测试均为独立脚本（直接 `python3` 运行，非 pytest），全部实测通过：

| 脚本 | 层次 | 内容 | 外部依赖 |
|------|------|------|---------|
| `tests/test_rtc_device.py` | 单元 | 重采样对齐、mic FIFO/增益/电平、spk 环形缓冲/欠载补零、viseme 调度、帧拷贝语义 | 无 |
| `tests/test_rtc_loopback.py` | 进程内回环 | 第二个 `RTCPeerConnection` 模拟浏览器（正弦波假 mic + 绿屏假摄像头），走完真实 offer/answer/ICE/DTLS，验证 mic 上行、TTS 下行、viseme、摄像头帧四条链路 | 无（走真实 UDP/ICE，耗时几秒） |
| `tests/test_rtc_signaling.py` | HTTP 端到端 | 8299 端口真实起 uvicorn，测 409（未启用 RTC）、完整握手、400（畸形 body） | `httpx`、本地端口 |

```bash
.venv/bin/python tests/test_rtc_device.py
.venv/bin/python tests/test_rtc_loopback.py
.venv/bin/python tests/test_rtc_signaling.py
```

## 6. 已知边界与待办

当前版本的明确限制（后续按需更新本节）：

- [ ] **仅局域网**：无 STUN/TURN，跨网段 / NAT 环境不可用
- [ ] **无 ICE 重启 / Trickle ICE**：网络切换（Wi-Fi ↔ 蜂窝）后只能靠前端整体重连
- [ ] **单 peer**：新连接顶掉旧连接，不支持多设备同时接入
- [ ] **无 RTCDataChannel**：控制信令复用 WebSocket `/ws/emma/face`（设计选择，
      但若未来要走数据通道需新增通路）
- [ ] **真实设备验证程度未知**：三层测试全部通过，但落地后无迭代记录，
      在真实手机/平板浏览器上的表现未经系统验证
- [ ] 前端设备页与 Live2D 预览同页，尚不能独立作为"纯设备"轻量页面

文档维护说明：本文件随 `rtc_device.py` / 信令端点 / 前端设备模式的行为变化同步更新；
上述待办项完成一项勾掉一项。
