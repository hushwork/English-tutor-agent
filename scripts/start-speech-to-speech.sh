#!/bin/bash
# speech-to-speech 启动 | 项目路径 | 重启安全
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=local

cd /home/ubuntu/English-tutor-agent
exec .venv/bin/speech-to-speech \
  --mode realtime --stt faster-whisper --faster_whisper_stt_model_name base \
  --llm_backend responses-api --responses_api_base_url http://127.0.0.1:8080/v1 \
  --responses_api_api_key local \
  --tts kokoro --ws_port 8765 --log_level INFO
