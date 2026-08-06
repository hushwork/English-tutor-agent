# 公网访问部署：frp + coturn + TLS 证书

> 把跑在**内网 NAT 后**的 Camera Tutor 暴露给公网（手机蜂窝网络实测可用）。
> 2026-08-06 落地，配套阅读：[WEBRTC.md](WEBRTC.md) §公网部署。

## 架构

```
手机（蜂窝/CGNAT）
  │  页面+信令  HTTPS :8200 ─┐
  │                          ├─► 云服务器（大陆，本例 124.221.233.118）
  │  音视频媒体  TURN :3478 ─┘        ├─ frps    :7000（frp 控制通道）
  │        （UDP 优先/TCP 兜底）       ├─ coturn  :3478 + relay 50000-50100/UDP
  │                                   └─（可选）frps 管理面板 :7500
  │                                          │ frp 隧道（frpc 主动外联，无需本地公网 IP）
  ▼                                          ▼
                                     内网 gpu-server
                                       ├─ dashboard :8200（页面+/rtc/offer+/rtc/config）
                                       └─ aiortc（WebRTC 媒体端）
```

要点：页面/信令走 frp **TCP** 隧道；WebRTC 媒体是 UDP 点对点，**穿不过 frp TCP**，
必须由 coturn 做 TURN 中转。两者可以不在同一台云服务器上（本例最初 TURN 在香港、
页面在大陆，后为延迟全部迁到大陆）。

## 1. frps（云服务器）

二进制：`/usr/local/bin/frps`（v0.69.1，从 [frp releases](https://github.com/fatedier/frp/releases) 下载对应架构）

`/etc/frp/frps.toml`：

```toml
bindPort = 7000

# 管理面板（可选，不用可整段删除并关闭 7500 防火墙）
webServer.addr = "0.0.0.0"
webServer.port = 7500
webServer.user = "admin"
webServer.password = "<换成强密码>"

# 认证令牌，frpc 必须匹配
auth.token = "<换成强 token>"

log.to = "console"
log.level = "info"
log.maxDays = 3
```

`/etc/systemd/system/frps.service`：

```ini
[Unit]
Description=FRP Server
After=network.target
Wants=network.target
[Service]
Type=simple
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.toml
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
[Install]
WantedBy=multi-user.target
```

启用：`sudo systemctl enable --now frps`

## 2. frpc（内网 gpu-server）

`/opt/frp/frpc.toml`（systemd 单元为 `frpc.service`，`nobody` 用户运行）：

```toml
serverAddr = "<云服务器IP>"
serverPort = 7000
auth.token = "<与 frps 相同>"
transport.tls.enable = true

# SSH 回连隧道：ssh -p 6000 <云服务器IP>
[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6000

# dashboard（页面+信令）
[[proxies]]
name = "tutor-dashboard"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8200
remotePort = 8200
```

改配置后 `sudo systemctl restart frpc`，日志看 `journalctl -u frpc`，
出现 `start proxy success` 才算隧道建立。

## 3. coturn（云服务器）

`apt install coturn`，`/etc/turnserver.conf`：

```
external-ip=<公网IP>/<内网IP>     # 关键！云主机公网 IP 是 1:1 NAT 映射，
                                 # 只写公网 IP 不生效，relay 会错误宣告内网地址
listening-port=3478
fingerprint
lt-cred-mech
user=tutor:<换成强密码>
realm=turn.local
min-port=50000
max-port=50100
log-file=stdout
```

`sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn`，
`sudo systemctl enable --now coturn`。

## 4. TLS 证书（内网 gpu-server）

远程浏览器 `getUserMedia` 要求安全上下文，mkcert 签发，**SAN 必须包含所有访问地址**：

```bash
cd certs
mkcert -cert-file tutor-dev.pem -key-file tutor-dev-key.pem \
  localhost 127.0.0.1 <内网IP> <云服务器IP>
```

`.env` 指向证书：

```
DASHBOARD_TLS_CERT=/home/ubuntu/workspace/english-tutor/certs/tutor-dev.pem
DASHBOARD_TLS_KEY=/home/ubuntu/workspace/english-tutor/certs/tutor-dev-key.pem
```

注意：

- 换/加公网 IP 后要**重新签发证书**并重启 realtime_demo（证书只在启动时加载）
- mkcert 根 CA 是自签的，新设备需安装根证书（`mkcert -CAROOT` 下的 `rootCA.pem`），
  否则浏览器仍会警告（可手动点继续）
- `certs/` 和 `.env` 都在 `.gitignore` 中，**不要提交**

## 5. .env 的 TURN 配置（内网 gpu-server）

```
# 本机网络出公网 UDP 被拦，TURN 走 TCP（浏览器端会自动补 UDP 变体）
RTC_TURN_URL=turn:<云服务器IP>:3478?transport=tcp
RTC_TURN_USER=tutor
RTC_TURN_PASS=<与 turnserver.conf 相同>
```

详见 [WEBRTC.md](WEBRTC.md) 配置项汇总。若本机网络不封 UDP，去掉 `?transport=tcp` 走 UDP 更佳。

## 6. 防火墙放行清单（云服务器控制台）

| 端口 | 协议 | 用途 |
|------|------|------|
| 7000 | TCP | frp 控制通道 |
| 8200 | TCP | 页面/信令入口 |
| 6000 | TCP | SSH 回连隧道（可选） |
| 7500 | TCP | frps 管理面板（可选） |
| 3478 | TCP+UDP | TURN 信令 |
| 50000-50100 | UDP | TURN 媒体中转 |

## 7. 验证

```bash
# 隧道
curl -sk https://<云服务器IP>:8200/api/health
# TURN relay candidate（在内网 gpu-server 上，.env 已配好）
.venv/bin/python - <<'EOF'
import asyncio, logging
logging.basicConfig(level=logging.CRITICAL)
from dotenv import load_dotenv; load_dotenv()
from aiortc import RTCPeerConnection
from camera_tutor.rtc_device import _rtc_configuration
async def main():
    pc = RTCPeerConnection(configuration=_rtc_configuration())
    pc.createDataChannel("t")
    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete": await asyncio.sleep(0.1)
    for l in pc.localDescription.sdp.splitlines():
        if l.startswith("a=candidate:"): print(l)
    await pc.close()
asyncio.run(main())
EOF
# 应看到: ... typ relay <云服务器公网IP> 500xx
```

手机实测：浏览器打开 `https://<云服务器IP>:8200/static/face_preview.html?device=1`，
agent 日志出现 `RTC connection state: connected` 即全链路通。

## 8. 运维备忘（踩过的坑）

- **重启 realtime_demo**：`pkill` 后旧进程优雅退出约需 9 秒，必须等死透
  （`pgrep -f realtime_demo` 无输出）再跑 `scripts/start-all.sh start-webrtc`，
  否则脚本误判"已在运行"跳过启动，服务实际没起来
- **网络封 UDP 的排查**：TURN over UDP 无响应时，双向 tcpdump 定位
  （云端 `tcpdump -nn 'udp port 3478'` + 本地 `tcpdump -i <网卡> 'udp and host <云IP>'`），
  本例实测家用宽带连公共 DNS 的 UDP 都被拦，只能 `?transport=tcp`
- **frpc 配置热更**：改完 `frpc.toml` 必须 `systemctl restart frpc`，
  且改 `serverAddr` 迁移 frps 时，防火墙 7000 要先放行，否则 tunnel 建立失败
- **dashboard 无鉴权**：公网暴露 8200 后任何人可访问页面。
  信令接口可用 `RTC_TOKEN` 保护；页面级防护目前依赖地址不公开，介意的话
  在云防火墙对 8200 做源 IP 限制
