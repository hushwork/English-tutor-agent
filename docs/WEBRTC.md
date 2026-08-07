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

接线点在 `camera_tutor/agent.py`：`av_source="webrtc"` 时运行时不直接持有
音视频对象，而是给 `RTCDeviceManager` 注册 session 钩子——每路浏览器 peer
接入即创建一个 `PracticeSession`（`camera_tutor/practice_session.py`），
拿到该 peer 自己的 `RTCAudioManager`/`RTCFrameSource` 后独立跑
VAD/ASR/LLM/TTS/视觉全链路。多用户并发时各会话完全隔离。

### 线程模型

- `RTCPeerConnection` 和所有 track I/O 都活在 **uvicorn 事件循环**里
  （`/rtc/offer` 端点在该循环中运行）。
- agent 的工作线程只接触带锁的 `deque`/`bytearray`，**没有跨事件循环调用**。

## 2. 信令

- 方式：**HTTP 一次性 offer/answer**（非 WebSocket）。
  浏览器 POST SDP offer 到 `POST /rtc/offer`（`dashboard_server.py:375`），同步返回 answer。
- 前端不 Trickle：先等 ICE gathering 完成再发 offer（`face_preview.html:98`）。
- ICE 配置：默认无 STUN/TURN（局域网直连）；公网部署时用 env 配置 TURN
  （`rtc_device.py: ice_servers()`），**浏览器端通过 `GET /rtc/config` 动态拉取同一份
  配置**，不在页面里硬编码。移动端浏览器在运营商 CGNAT 后只有内网 candidate，
  不配 TURN 必然打不通（2026-08-06 实锤）。
- 可选鉴权：环境变量 `RTC_TOKEN` 设置后，`/rtc/offer` 和 `/rtc/config` 均要求
  `Authorization: Bearer <token>`；页面侧用 `?token=xxx` 传递。
- **多用户并发**（2026-08-07 起）：每个 offer 创建独立的 `RTCSession`
  （自己的 peer connection + 音频/视频替身），互不干扰；同一 `user_id`
  重复连接只替换该用户自己的旧 session。offer 请求体带 `user_id`
  （`face_preview.html?user=xxx`，非法 ID 返回 400，缺省为 `default`），
  应答带 `session_id`/`user_id`。学习数据按用户隔离在
  `.camera-tutor-data/{user_id}/`；viseme/摄像头画面经 `/ws/emma/face?user=xxx`
  按用户路由。并发规模受模型后端限制：本地 llama-server 用
  `LLAMA_PARALLEL`（启动脚本 `-np`，默认 4 槽）调并发槽，单卡实际 2–4 人；
  更多人走云端 MaaS（不设 `OMNI_WS_URL`）。
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

硬件降级链（2026-08-07）：音视频 → **纯语音**（无摄像头时自动降级，对话不受影响）→
**仅预览**（连麦克风都没有时提供入口，可看 Live2D/字幕，不建 RTC）。

ICE 等待有 4 秒超时兜底：服务端不做 trickle，浏览器端等 gathering complete
才发 offer，但配了 TURN 而中继不可达/过慢时 gathering 可能永远不结束
（表现为"点击开始没反应"）；超时后拿已有候选直接发 offer——LAN 下 host
candidate 本就够用。

### 麦克风采集开关

dashboard → Device 标签页 →「🎤 Mic Capture (WebRTC)」可切换三项
getUserMedia 音频处理（`GET/POST /api/device/audio-settings`，持久化在
`.camera-tutor-data/audio_settings.json`，设备端下次采集时生效）：

| 开关 | 默认 | 说明 |
|------|------|------|
| `echoCancellation` (AEC) | 开 | 外放场景防回采 |
| `noiseSuppression` | 关 | 会把小声语音当噪音吃掉 |
| `autoGainControl` (AGC) | 关 | 放大噪音底，破坏服务端 RMS VAD |

### 配置项汇总

| 变量 | 默认 | 说明 |
|------|------|------|
| `AV_SOURCE` | `local` | `webrtc` 启用远程设备模式 |
| `VISME_LEAD_MS` | 80 | 唇形同步补偿（毫秒） |
| `RTC_TOKEN` | 空（不鉴权） | `/rtc/offer`、`/rtc/config` 的 Bearer 令牌 |
| `DASHBOARD_TLS_CERT` / `DASHBOARD_TLS_KEY` | 空 | HTTPS 证书（远程浏览器必需） |
| `RTC_TURN_URL` / `RTC_TURN_USER` / `RTC_TURN_PASS` | 空 | 公网穿透的 TURN 中转（coturn）。支持 `?transport=tcp` 后缀 |
| `RTC_STUN_URL` | 空 | STUN（可选，一般有 TURN 即可） |

### 公网部署（frp + coturn）

> 完整的 frps/frpc/coturn/证书配置与运维备忘见 [PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)。

内网穿透的完整姿势（2026-08-06 落地）：

1. **页面/信令**：frps 架在公网服务器，本地 frpc 把 8200 映射出去
   （TCP 隧道即可，WebRTC 信令是 HTTP）
2. **媒体**：coturn 架在**同一台或另一台**公网服务器，
   `.env` 配 `RTC_TURN_URL=turn:<ip>:3478` + 账号密码，浏览器和 agent 自动共用
3. **证书**：mkcert 签发的证书 SAN 要包含公网 IP，否则浏览器报域名不匹配
4. coturn 在云主机（公网 IP 是 NAT 映射）上必须配
   `external-ip=<公网IP>/<内网IP>`，只写公网 IP 不生效
   （relay candidate 会错误地宣告内网地址）
5. **网络封 UDP 的环境**（实测某些家用宽带出公网 UDP 全被拦，连 DNS 都不通）：
   `RTC_TURN_URL` 加 `?transport=tcp`。浏览器端不受影响——`/rtc/config`
   下发时会自动补 UDP 变体（UDP 优先、TCP 兜底，见 `browser_ice_servers()`）
6. TURN relay 端口段（默认配置 50000-50100 UDP）和 3478 TCP/UDP
   都要在云防火墙/安全组放行

## 5. 测试

测试均为独立脚本（直接 `python3` 运行，非 pytest），全部实测通过：

| 脚本 | 层次 | 内容 | 外部依赖 |
|------|------|------|---------|
| `tests/test_rtc_device.py` | 单元 | 重采样对齐、mic FIFO/增益/电平、spk 环形缓冲/欠载补零、viseme 调度、帧拷贝语义 | 无 |
| `tests/test_rtc_loopback.py` | 进程内回环 | 第二个 `RTCPeerConnection` 模拟浏览器（正弦波假 mic + 绿屏假摄像头），走完真实 offer/answer/ICE/DTLS，验证 mic 上行、TTS 下行、viseme、摄像头帧四条链路 | 无（走真实 UDP/ICE，耗时几秒） |
| `tests/test_rtc_multiuser.py` | 多用户并发 | 两用户同时接入不互踢、会话间音频隔离、同用户重连只换自己、断连只清理自己、user_id 校验 | 无 |
| `tests/test_rtc_signaling.py` | HTTP 端到端 | 8299 端口真实起 uvicorn，测 409（未启用 RTC）、完整握手、400（畸形 body） | `httpx`、本地端口 |
| `tests/test_rtc_browser.py` | **真实浏览器** | headless Chrome（假麦克风/摄像头设备）打开真实 `face_preview.html?device=1` 页面并点击开始，验证：页面连接状态、服务端 peer、mic 上行有声、摄像头帧、浏览器 RTP 统计（TTS 下行字节 > 0） | `playwright` + 系统 Chrome |

```bash
.venv/bin/python tests/test_rtc_device.py
.venv/bin/python tests/test_rtc_loopback.py
.venv/bin/python tests/test_rtc_multiuser.py
.venv/bin/python tests/test_rtc_signaling.py
.venv/bin/python tests/test_rtc_browser.py   # 最接近真实设备的自动化验证
```

### 物理设备手动验证清单

真实手机/平板上的验证步骤（自动化无法覆盖的部分：硬件权限弹窗、真实麦克风回声消除、移动网络抖动）：

1. 配置 TLS（`DASHBOARD_TLS_CERT/KEY`，mkcert 签发并在手机上信任根证书）
2. 启动：`.venv/bin/python camera_tutor/realtime_demo.py --av-source webrtc`
3. 手机浏览器打开 `https://<agent-ip>:8200/static/face_preview.html?device=1`，点开始
4. 逐项确认：
   - [ ] 权限弹窗正常授予麦克风+摄像头
   - [ ] 页面显示"🟢 设备已连接"，RTC 指示灯亮
   - [ ] 对手机说话，agent 日志出现 STT 结果
   - [ ] 手机扬声器能听到 Emma 回复，Live2D 唇形基本同步
   - [ ] 锁屏/切后台后回前台，2 秒内自动重连恢复
   - [ ] Wi-Fi 信号弱/切换网络时的表现（当前预期：断线重连，可能需重开页面）

## 6. 已知边界与待办

当前版本的明确限制（后续按需更新本节）：

- [x] ~~**仅局域网**：无 STUN/TURN，跨网段 / NAT 环境不可用~~
      **已支持 TURN 中转**（2026-08-06，`RTC_TURN_*` env + `/rtc/config` 下发，
      手机蜂窝 CGNAT 环境实测连通）
- [ ] **无 ICE 重启 / Trickle ICE**：网络切换（Wi-Fi ↔ 蜂窝）后只能靠前端整体重连
- [x] ~~**单 peer**：新连接顶掉旧连接，不支持多设备同时接入~~
      **已支持多用户并发**（2026-08-07）：每 offer 一个独立 `RTCSession` +
      `PracticeSession`，学习数据按 `user_id` 隔离；本地 llama-server 并发槽
      用 `LLAMA_PARALLEL` 调（单卡实际 2–4 人，更多人走云端 MaaS）
- [ ] **无 RTCDataChannel**：控制信令复用 WebSocket `/ws/emma/face`（设计选择，
      但若未来要走数据通道需新增通路）
- [x] ~~真实设备验证程度未知~~ **自动化真实浏览器验证已通过**（2026-08-05，
      `tests/test_rtc_browser.py`：真实 Chrome + 真实页面全链路通过）；
      物理手机手动验证已通过（2026-08-06，蜂窝网络 + TURN 中转全链路）
- [ ] 前端设备页与 Live2D 预览同页，尚不能独立作为"纯设备"轻量页面

### 已修复问题记录（2026-08-07 设备页交互）

- **点击"开始对话"无反应**：ICE gathering 等待无超时，TURN 中继分配慢/不可达时
  永久卡在 gathering、offer 发不出去。修复：4 秒超时兜底 + 点击即显示"连接中"。
- **无硬件设备被遮罩困死**：getUserMedia 失败后只有重试一条路。修复：硬件降级链
  （音视频 → 纯语音 → 仅预览）。
- **开始前无法进设置**：全屏遮罩盖住设置入口。修复：导师徽章/设置按钮 z-index
  提到遮罩之上，开始对话前即可进 dashboard 换导师、调采集开关。
- **Live2D `_subdelegates is null`**：页面卸载瞬间渲染循环残留帧回调访问已释放
  实例（SDK 示例代码竞态，无害但刷红错）。修复：`live2d-src/lappdelegate.ts`
  源码加防御（下次 `npm run build` 生效）；bundle 重建前由 `face_preview.html`
  应用层精准屏蔽该报错。

### 已修复问题记录（2026-08-06 公网部署）

- **服务器在 NAT 后，host candidate 不可达**：远程浏览器只能拿到服务器的内网
  candidate，ICE 全部失败。修复：coturn TURN 中转（`RTC_TURN_*` env 注入 aiortc）。
- **手机在运营商 CGNAT 后**：浏览器无 STUN/TURN 时只报 `10.x` 内网 candidate，
  服务器 relay 也路由不到。修复：`GET /rtc/config` 给浏览器下发同一份 ICE 配置。
- **家用宽带封死出公网 UDP**（连公共 DNS UDP 都不通）：TURN over UDP 无响应。
  修复：`RTC_TURN_URL` 用 `?transport=tcp`；浏览器端自动补 UDP 变体。
- **coturn relay 宣告内网地址**：云主机公网 IP 是 1:1 NAT 映射，
  `external-ip` 只写公网 IP 不生效，必须 `external-ip=<公网>/<内网>` 显式映射。
- **移动端页面未适配**：状态栏溢出、摄像头窗过大。修复：`face_preview.html`
  加 `@media (max-width: 600px)` 适配。
- **部署注意**：`pkill` realtime_demo 后要**等旧进程完全退出**（优雅退出约 9 秒）
  再跑 `start-all.sh`，否则脚本 pgrep 误判"已在运行"而跳过启动，服务实际没起来。

### 已修复问题记录（2026-08-05 真机联调）

- **双重 offer 竞态**：页面重连发两次 offer 时，旧 peer 的迟到收尾事件会把新连接的
  `_peer_connected` 清掉——mic 正常但 `write_spk` 全部丢弃（对端无声）。
  修复：旧连接的 track 收尾 / connectionstatechange 事件按身份忽略；
  `connected` 状态时复位标志（`rtc_device.py`）。
- **唇形不动**：开 TLS 后 `face_sync.py` 硬编码的 `ws://`/`http://` 推送全部失败
  （WS 握手失败 + HTTP fallback 端点不存在且异常被静默吞掉）。
  修复：按 `DASHBOARD_TLS_CERT/KEY` 自动切换 `wss/https`。
- **浏览器 DSP 破坏 ASR**：Chrome 默认 AGC 把噪音底放大到 RMS VAD 阈值之上
  （whisper 对着噪音幻觉出完整句子），`noiseSuppression` 会把小声语音当噪音吃掉。
  修复：`getUserMedia` 显式 `autoGainControl: false, noiseSuppression: false`，
  保留 `echoCancellation` 防外放回采；服务端配合 `MIC_GAIN` 数字增益补偿 +
  `VAD_THRESHOLD` 按干净噪音底下调。
- **部署注意**：dashboard 开 TLS 后，本机健康检查等硬编码 `http://localhost:8200`
  的调用会失败——端口被残留实例占用时新实例的 uvicorn 绑定失败退出，
  排查先看 `ss -tlnp | grep 8200` 确认没有旧进程。

文档维护说明：本文件随 `rtc_device.py` / 信令端点 / 前端设备模式的行为变化同步更新；
上述待办项完成一项勾掉一项。
