#!/bin/bash
# =============================================================
#  Gemma 4 E2B safetensors → GGUF 转换 + 量化脚本
# =============================================================
set -e

MODEL_DIR="/home/ubuntu/models/gemma4-e2b"
CONVERT_DIR="/tmp/llama-convert"
LLAMA_DIR="/home/ubuntu/English-tutor-agent/bin/llama-b10223"

cd "$CONVERT_DIR"
export PYTHONPATH="$CONVERT_DIR:$CONVERT_DIR/gguf:$PYTHONPATH"
export NO_LOCAL_GGUF=1  # 使用 pip 安装的 gguf 包

echo "=== Step 1: safetensors → FP16 GGUF ==="
python3 convert_hf_to_gguf.py \
  "$MODEL_DIR" \
  --outfile "$MODEL_DIR/gemma-4-E2B-it-F16.gguf" \
  --outtype f16

echo ""
echo "=== Step 2: 提取 mmproj（视觉编码器）==="
python3 convert_hf_to_gguf.py \
  "$MODEL_DIR" \
  --mmproj "$MODEL_DIR/mmproj-gemma-4-E2B-it-f16.gguf"

echo ""
echo "=== Step 3: Q4_K_M 量化 ==="
"$LLAMA_DIR/llama-quantize" \
  "$MODEL_DIR/gemma-4-E2B-it-F16.gguf" \
  "$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf" \
  Q4_K_M

echo ""
echo "=== 完成 ==="
ls -lh "$MODEL_DIR"/*.gguf
echo ""
echo "启动命令: bash scripts/start-gemma.sh"
