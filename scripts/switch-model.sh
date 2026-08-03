#!/bin/bash
# 切换 Gemma 模型: switch-model.sh e2b|e4b
M=${1:-e4b}
case $M in
  e2b) M_PATH=/home/ubuntu/models/gemma4-e2b/gemma-4-E2B-it-Q4_K_M.gguf ;;
  e4b) M_PATH=/home/ubuntu/models/gemma4-e4b/gemma-4-E4B-it-Q4_K_M.gguf ;;
  *) echo "用法: $0 e2b|e4b"; exit 1 ;;
esac

SIMPLE_CT='{% for m in messages %}{% if m.role == "user" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% elif m.role == "assistant" %}<start_of_turn>model\n{{ m.content }}<end_of_turn>\n{% elif m.role == "system" %}<start_of_turn>user\n{{ m.content }}<end_of_turn>\n{% endif %}{% endfor %}<start_of_turn>model\n'

kill $(pgrep llama-server) 2>/dev/null
sleep 2
echo "启动 $M 到 :8080..."
"$(dirname "${BASH_SOURCE[0]}")/../bin/llama-b10223"/llama-server \
  -m "$M_PATH" --host 127.0.0.1 --port 8080 \
  -c 4096 --no-webui --n-gpu-layers all \
  --no-jinja --chat-template "$SIMPLE_CT" &
sleep 6
echo "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "✅ $M 就绪"
