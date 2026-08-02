#!/bin/bash
# 一键启动全链路（重启安全，所有文件在项目目录内）
set -e
cd /home/ubuntu/English-tutor-agent
mkdir -p logs

LLAMA=bin/llama-b10223/llama-server
MODEL=$1

case "${MODEL:-qwen}" in
  qwen)
    M_FLAG="-m /home/ubuntu/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj /home/ubuntu/models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    CT_FLAG=""
    ;;
  e2b)
    M_FLAG="-m /home/ubuntu/models/gemma4-e2b/gemma-4-E2B-it-Q4_K_M.gguf"
    CT_FLAG="--no-jinja --chat-template '{% for m in messages %}{% if m.role == \"user\" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% elif m.role == \"assistant\" %}<start_of_turn>model\n{{ m.content }}<end_of_turn>\n{% elif m.role == \"system\" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% endif %}{% endfor %}<start_of_turn>model\n'"
    ;;
  e4b)
    M_FLAG="-m /home/ubuntu/models/gemma4-e4b/gemma-4-E4B-it-Q4_K_M.gguf"
    CT_FLAG="--no-jinja --chat-template '{% for m in messages %}{% if m.role == \"user\" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% elif m.role == \"assistant\" %}<start_of_turn>model\n{{ m.content }}<end_of_turn>\n{% elif m.role == \"system\" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% endif %}{% endfor %}<start_of_turn>model\n'"
    ;;
  *) echo "用法: $0 qwen|e2b|e4b"; exit 1 ;;
esac

echo "启动 $MODEL ..."
pkill -f "llama-server" 2>/dev/null; sleep 2
eval exec $LLAMA $M_FLAG --host 127.0.0.1 --port 8080 -c 4096 --no-webui --n-gpu-layers all $CT_FLAG > logs/llama.log 2>&1 &
sleep 6
echo "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "✅ $MODEL :8080"
