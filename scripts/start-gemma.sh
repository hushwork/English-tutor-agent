#!/bin/bash
# =============================================================
#  Gemma 4 E2B 方案 — 启动本地推理服务
#  特点：文本 + 图像 + 音频 原生输入（跳过 STT）
# =============================================================
set -e

MODEL_DIR="/home/ubuntu/models/gemma4-e2b"
GGUF="$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-gemma-4-E2B-it-f16.gguf"

# 如果找不到 Q4_K_M，尝试 FP16
if [ ! -f "$GGUF" ]; then
    GGUF="$MODEL_DIR/gemma-4-E2B-it-F16.gguf"
fi

echo "Gemma 4 E2B 启动中..."
echo "  模型: $GGUF"
echo "  视觉: $MMPROJ"

exec "$(dirname "${BASH_SOURCE[0]}")/../bin/llama-b10223"/llama-server \
  -m "$GGUF" \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --host 127.0.0.1 --port 8081 \
  -c 8192 \
  --no-webui \
  --n-gpu-layers all
