# 本地化部署完成状态

> 更新日期：2026-08-02

## 当前架构

```
realtime_demo.py ──ws──▶ local_pipe.py (:8765) ──▶ llama-server (:8080)
                            │
              VAD → whisper-base STT → LLM → Kokoro TTS
```

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

```bash
# 1. 启动 LLM
bash scripts/start-llm.sh e4b

# 2. 启动语音管线
CUDA_VISIBLE_DEVICES="" nohup .venv/bin/python3 scripts/local_pipe.py > logs/pipe.log 2>&1 &

# 3. 运行
python camera_tutor/realtime_demo.py --select-devices
```

## 已知问题

1. **Gemma E4B 视觉**：不支持跨轮视觉记忆，每轮独立看图
2. **Gemma E4B 旁白**：偶尔生成 `(描述动作)` 括号内容
3. **STT 精度**：whisper-base 对儿童口音识别率有限
4. **模型切换**：换模型需重启 llama-server 和 pipe

## 回退云端

修改 `.env`：`DEPLOY_MODE=cloud`，取消注释云端 API 配置即可。
