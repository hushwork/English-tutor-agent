#!/bin/bash
# 一键启动/停止 Camera Tutor 全链路（重启安全）
# 用法: scripts/start-all.sh [start|stop|status]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="$(pwd)/.venv/bin/python"
LLAMA=bin/llama-b10223/llama-server
MODELS=/home/ubuntu/models
LOGS=logs
mkdir -p "$LOGS"

# Gemma chat template（与 scripts/start-llm.sh 一致）
CT='{% for m in messages %}{% if m.role == "user" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% elif m.role == "assistant" %}<start_of_turn>model\n{{ m.content }}<end_of_turn>\n{% elif m.role == "system" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% endif %}{% endfor %}<start_of_turn>model\n'

wait_port() {  # wait_port <port> <name>
  for _ in $(seq 1 30); do
    ss -tln | grep -q ":$1 " && { echo "✅ $2 :$1"; return 0; }
    sleep 2
  done
  echo "⚠️  $2 :$1 启动超时，查看 $LOGS/"
  return 1
}

start() {
  # WebRTC 远程设备模式：.env 里 AV_SOURCE=webrtc 或 `start-all.sh start-webrtc`
  local webrtc=0
  [ "${1:-}" = "webrtc" ] && webrtc=1
  grep -q '^AV_SOURCE=webrtc' .env 2>/dev/null && webrtc=1

  # 1) llama-server: Gemma-4-E4B + mmproj 视觉（GPU 卸载；注意本机 Vulkan 显存
  #    上报偶发损坏，若启动即崩可改回 --device none 强制 CPU）
  if ! ss -tln | grep -q ':8080 '; then
    nohup $LLAMA -m $MODELS/gemma4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
      --mmproj $MODELS/gemma4-e4b/mmproj-BF16.gguf \
      --host 127.0.0.1 --port 8080 -c 8192 -t 12 --no-webui --n-gpu-layers all \
      --no-jinja --chat-template "$CT" > $LOGS/llama.log 2>&1 &
  fi
  wait_port 8080 llama-server || return 1

  # 2) 家长仪表盘（独立进程，不随 demo 重启而死）
  #    WebRTC 模式下跳过：dashboard 必须由 realtime_demo 在同进程拉起（/rtc/offer 信令）
  if [ "$webrtc" = 0 ] && ! ss -tln | grep -q ':8200 '; then
    nohup $PY -m uvicorn camera_tutor.dashboard_server:app \
      --host 0.0.0.0 --port 8200 --log-level warning > $LOGS/dashboard.log 2>&1 &
    wait_port 8200 dashboard || return 1
  fi

  # 3) 本地语音管道（whisper STT → LLM → Kokoro TTS）
  if ! ss -tln | grep -q ':8765 '; then
    nohup $PY scripts/local_pipe.py > $LOGS/local_pipe.log 2>&1 &
  fi
  wait_port 8765 local_pipe || return 1

  # 4) 主程序（设备选择自动从 .camera-tutor-data/devices.json 按名字恢复）
  if ! pgrep -f 'camera_tutor/realtime_demo.py' > /dev/null; then
    local av_args=""
    [ "$webrtc" = 1 ] && av_args="--av-source webrtc"
    nohup $PY camera_tutor/realtime_demo.py $av_args > $LOGS/realtime_demo.log 2>&1 &
    sleep 10
  fi
  echo "✅ realtime_demo 已启动"
  if [ "$webrtc" = 1 ]; then
    echo "   📱 设备端: https://$(hostname -I | awk '{print $1}'):8200/static/face_preview.html?device=1"
  else
    echo "   Emma 形象: http://$(hostname -I | awk '{print $1}'):8200/static/face_preview.html"
  fi
}

stop() {
  pkill -f 'camera_tutor/realtime_demo.py' 2>/dev/null
  PID=$(ss -tlnp 2>/dev/null | grep ':8765 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | head -1)
  [ -n "$PID" ] && kill "$PID" 2>/dev/null
  PID=$(ss -tlnp 2>/dev/null | grep ':8200 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | head -1)
  [ -n "$PID" ] && kill "$PID" 2>/dev/null
  pkill -x llama-server 2>/dev/null
  sleep 2
  echo "🛑 已全部停止"
}

status() {
  ss -tln | grep -q ':8080 ' && echo "✅ llama-server :8080" || echo "❌ llama-server"
  ss -tln | grep -q ':8200 ' && echo "✅ dashboard    :8200" || echo "❌ dashboard"
  ss -tln | grep -q ':8765 ' && echo "✅ local_pipe   :8765" || echo "❌ local_pipe"
  pgrep -f 'camera_tutor/realtime_demo.py' > /dev/null && echo "✅ realtime_demo" || echo "❌ realtime_demo"
}

case "${1:-start}" in
  start)  start ;;
  start-webrtc) start webrtc ;;
  stop)   stop ;;
  status) status ;;
  *) echo "用法: $0 [start|start-webrtc|stop|status]"; exit 1 ;;
esac
