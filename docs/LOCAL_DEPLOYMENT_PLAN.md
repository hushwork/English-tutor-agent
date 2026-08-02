# English Tutor / Camera Tutor — 本地化部署实施计划

> 日期：2026-08-01
> 范围：把现有依赖云端 API 的调用链全部本地化（隐私 / 离线可用 / 零 API 成本）
> 主功能入口：`camera_tutor/realtime_demo.py`（实时语音对话）+ `english_tutor/web_server.py`（Web UI）
> 硬件档位：A = RTX 4060 Ti **8GB**（当前实测），B = **16GB**（升级目标，如 4060 Ti 16GB / 3060 12GB+）

---

## 1. 现状盘点：现有云端调用点（代码级）

| # | 文件 | 用途 | 协议 / 服务 | 关键配置 |
|---|------|------|-------------|----------|
| 1 | `english_tutor/llm_client.py` | CLI/Web 纯文本对话 | OpenAI 兼容 REST `/chat/completions` → DeepSeek/MaaS | `LLM_BASE_URL`、`LLM_MODEL`、`DEEPSEEK_API_KEY` |
| 2 | `camera_tutor/omni_client.py` | 摄像头视觉分析 + 文本生成 | OpenAI 兼容 REST → DashScope/MaaS（`qwen3.5-omni-flash-realtime`） | `DASHSCOPE_API_KEY`、`OMNI_CLOUD_MODEL`、`LLM_BASE_URL` |
| 3 | `camera_tutor/agent.py` + `connection.py` | **实时语音对话（主链路）** | **OpenAI Realtime 协议 WebSocket** → 阿里 MaaS `wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime` | `DASHSCOPE_API_KEY`、`WORKSPACE_ID`（agent.py L101-104 构造 `ws_url`） |
| 4 | `camera_tutor/demo.py` | 备用演示入口 | `OmniClient(mode=CLOUD)` | 同上 |
| 5 | `english_tutor/tts.py`、`web_server.py` L217/552/636 | 语音合成 | **edge-tts（微软云端）** | 无 key，但依赖网络 |
| 6 | `english_tutor/stt.py` | 语音识别 | **已本地** faster-whisper（`tiny`） | — |

> 关键发现：主链路（#3）走的是 **OpenAI Realtime WebSocket 协议**（`session.update` / `server_vad` / `response.cancel` / `audio.delta`，见 agent.py L364-377），不是普通 REST。这决定了本地替代方案必须提供同协议的服务端（见 §6）。

---

## 2. 模型选型：量化说明

Ollama 官方 tag 默认即 **GGUF Q4_K_M 量化**（质量损失 <1%，8/16GB 显存部署的标准做法）：

| 模型 | 原始 FP16 | Q4_K_M 量化后 |
|------|-----------|---------------|
| qwen3-vl:8b | ~16GB | **6.1GB** |
| qwen3-vl:4b | ~8GB | **3.3GB** |
| qwen3:8b（纯文本） | ~16GB | **5.2GB** |
| gemma4:e2b | ~15GB | **7.2GB** |
| gemma4:e4b | ~19GB | **9.6GB** |
| gemma3:4b | ~8GB | **3.3GB** |

原则：8GB 档位统一用 Q4_K_M；16GB 档位可选 qwen3-vl:8b 的 Q8_0（~8.5GB，更准）或 gemma4:e4b。

---

## 3. qwen3-omni vs qwen3-vl 对比（决策依据）

| 维度 | **Qwen3-Omni** | **Qwen3-VL** |
|------|----------------|--------------|
| 尺寸 | **只有 30B-A3B（MoE）一个**，无小尺寸 | 2B / 4B / 8B / 32B + 30B-A3B MoE |
| Q4 量化后 | **~17.3GB + mmproj 1.2GB ≈ 18.5GB** | 4b≈3.3GB，8b≈6.1GB |
| 8GB 显存 | ❌ 彻底装不下 | ✅ 4b 舒适 / 8b 可跑（余 ~1.9GB） |
| 16GB 显存 | ❌ 需 CPU offload 慢跑 或 IQ2 2bit（质量崩） | ✅ 8b 舒适，可上 Q8_0 |
| 能力 | any-to-any：图像/音频输入 + **文本/语音输出**，端到端低延迟、可打断 | image-text-to-text：图像/视频→文本，**无音频输入输出** |
| ollama 官方支持 | ❌ 无官方库（404） | ✅ 官方库，全尺寸 tags |
| 结论 | **本硬件档位不可行** | **本地化唯一现实选择** |

**结论**：8/16GB 显存下不是选择题 —— Qwen3-Omni 装不下。本地语音对话必须走**组件化管线**：`VAD → STT(本地) → LLM(Qwen3-VL, 本地) → TTS(本地)`。这与项目 TECH_STACK_RATIONALE.md 中"Omni 优先，必要时混搭"的策略一致，只是硬件约束强制走混搭路线。

---

## 4. gemma4 vs qwen3-vl 路线对比

| 维度 | **Gemma 路线** | **Qwen3 路线（推荐）** |
|------|----------------|------------------------|
| 8GB 档位 | gemma4:e2b 7.2GB —— **权重即占满，KV cache/视觉编码器无余量，ctx 必须压到 4-8K，大概率 CPU offload 掉速**；gemma3:4b 3.3GB 舒适但是 2025-03 老模型 | qwen3-vl:8b 6.1GB，余 ~1.9GB，8K ctx 可用，视觉+对话一体 |
| 16GB 档位 | gemma4:e4b 9.6GB（多模态 文本+图像+音频）舒适 | qwen3-vl:8b Q8_0 舒适；或 qwen3-vl:4b+32b 升级路径 |
| 视觉能力 | E2B/E4B 原生多模态 | Qwen3-VL 2025-10 代际，OCR/图表/空间理解当前 4-8B 级第一梯队 |
| 生态 | ollama 官方支持 | ollama 官方支持 |
| 适用 | 对 Google 生态有硬性要求时 | **默认选择** |

**推荐**：**Qwen3 路线，主模型 `qwen3-vl:8b`**（8GB 与 16GB 两档通用，一个模型同时承担对话 + 摄像头视觉）。Gemma 路线留作备选（8GB 上 e2b 需实测确认不掉速才可用）。

### 4.1 Gemma 4 的独有价值：多模态综合输入（进阶路线，方案 B+）

Gemma 4 E2B/E4B 是 **any-to-text**：同一轮对话可同时输入 **文本 + 图像 + 音频**，跨模态综合理解（如"画面里孩子指着玩具 + 语音'this!' + 上文"联合推理），输出仍为文本。

- **价值**：音频**不经过 STT** 直接进模型 —— 儿童口音、语调、情绪不丢失（项目 TECH_STACK_RATIONALE.md"绕过 ASR"论点）；视觉+听觉联合推理。这是相比"qwen3-vl + STT"管线的**结构性差异**（提升理解准确度，非延迟/音频质量）。
- **约束**：8GB 档不可行（E2B 7.2GB + 音频编码器 + KV cache 溢出）；16GB 档可用（e4b 全显存）。
- **集成**：需 vLLM 部署（官方支持 Gemma4 E2B/E4B 含音频；Ollama audio 输入支持弱）+ 新增"音频直通"代码路径（绕过 STT）；**huggingface/speech-to-speech 不支持此模式**，实时对话层需另做。
- **定位**：仅在 16GB 硬件 + 实测儿童口音在 STT 环节丢失严重时启动 PoC。

---

## 5. 推荐架构

```
┌────────────────────────────────────────────────────────────────┐
│  Camera Tutor (realtime_demo.py) + Web UI (web_server.py)       │
└───────────────┬───────────────────────────────┬────────────────┘
                │ OpenAI Realtime WS (不改协议)  │ OpenAI 兼容 REST
                ▼                               ▼
┌───────────────────────────────┐   ┌──────────────────────────┐
│ huggingface/speech-to-speech  │   │   Ollama  (localhost:11434)│
│ ws://localhost:8765/v1/realtime│   │  /v1/chat/completions     │
│  Silero VAD → STT → LLM → TTS │──▶│   主模型 qwen3-vl:8b      │
│  （LLM 槽位指向 Ollama）       │   │  （对话 + 视觉分析共用）   │
└───────────────────────────────┘   └──────────────────────────┘
   STT: faster-whisper small (CPU)        ▲
   TTS: Kokoro-82M (CPU, ~0.3GB)          │ 摄像头帧 → 视觉分析
                                          │ (omni_client → ollama)
```

- **GPU 内存预算（8GB 档）**：qwen3-vl:8b 权重 6.1GB + KV cache(8K ctx) ~0.7GB ≈ 7GB 常驻 GPU；STT/TTS/VAD 全部跑 CPU。余量 ~1GB，需小图（360×360，项目已默认）+ 适度 ctx。
- **16GB 档**：同一架构从容运行，可把 LLM 换 `qwen3-vl:8b` Q8_0 或 `gemma4:e4b`，TTS 可升级 Qwen3-TTS 1.7B。
- **收益**：视频帧、音频、对话内容全部不出本机；断网可用；API 费用归零（仅电费）。
- **引擎可插拔**：Ollama 与 vLLM 均提供 OpenAI 兼容 `/v1/chat/completions`，切换只改 `LLM_BASE_URL`，架构不绑定具体引擎。

### 5.1 推理引擎选型（Ollama / vLLM / llama.cpp）

**先澄清一个常见误区**：推理质量由模型本身决定，与引擎无关（同模型同量化下输出一致）。引擎只影响速度 / 吞吐 / 显存开销 / 功能，选型取决于场景。

| 维度 | **Ollama / llama.cpp** | **vLLM** |
|------|------------------------|----------|
| 优势场景 | **单机单用户、低延迟交互、8GB 小卡** | 高并发吞吐、生产服务、多用户 |
| 8GB 显存跑 7-8B VLM | ✅ 内存效率高，qwen3-vl:8b 的 6.1GB GGUF 直接跑 | ⚠️ 极紧：需 AWQ/GPTQ 4bit（~5GB）+ 严格 `--max-model-len`，KV cache 仅剩 ~1.5GB；**视觉 token 多会很快吃光 KV cache** |
| 量化格式 | GGUF 原生（Q4_K_M 标准） | GGUF 为"高度实验性"插件（官方文档警告，VLM GGUF 无官方验证）；应改用 AWQ/GPTQ |
| Qwen3-VL 支持 | ✅ | ✅ 2025-09 起官方支持，成熟 |
| Gemma4 支持 | ✅ | ✅ 2026-04 起支持（E2B/E4B/12B） |
| OpenAI 兼容 API | ✅ `/v1/chat/completions` | ✅ 更全（另有 `/v1/responses`） |
| 启动/部署 | 秒级、一行命令 | 分钟级、需模型转换 |
| speech-to-speech 对接 | ✅ base_url 指 Ollama | ✅ base_url 指 vLLM |

**本项目决策**：
- **8GB 档（当前硬件）→ Ollama**：显存效率高、VLM GGUF 原生、启动快，是 8GB 小卡的正确工具；vLLM 的并发吞吐优势在单用户场景用不上。
- **16GB 档 / 未来多用户或生产化 → vLLM（可选升级）**：Qwen3-VL 官方支持成熟，配合 AWQ/GPTQ 4bit 量化，`LLM_BASE_URL` 一行切换，架构无需改动。
- llama.cpp 裸用（`llama-server`）介于两者之间：无 Ollama 的模型管理便利，但可定制程度最高；**不作为本项目默认项**。

---

## 6. huggingface/speech-to-speech 集成说明

仓库：https://github.com/huggingface/speech-to-speech （PyPI 包 `speech-to-speech`，Apache-2.0）

**为什么选它**：项目主链路（§1 #3）是 OpenAI Realtime 协议客户端；该仓库恰好提供 **OpenAI Realtime 兼容的 WebSocket 服务**（`ws://localhost:8765/v1/realtime`），协议天然匹配 —— `connection.py` 只需换 URL 和 key，重连/回调/打断逻辑全部复用。

**组件与支持情况**：
- VAD：Silero VAD v5（内置）
- STT：默认 Parakeet TDT 0.6B；支持 **faster-whisper**（项目已有依赖，推荐，统一 STT 栈）
- LLM：默认 OpenAI Responses API；**可配任意 OpenAI 兼容服务器**（vLLM / llama.cpp / **Ollama**），也可直接 `--enable_llm_proxy` 把 LLM 暴露成 `/v1/chat/completions`
- TTS：默认 Qwen3-TTS 1.7B（GGML 量化 ~1-2GB）；**支持 Kokoro-82M**（英语教学场景推荐，~0.3GB CPU 可跑）
- 实时流式 + 说话打断（barge-in）：开箱即用（`server_vad`、`response.cancel`、`interrupt_response`）
- 部署形态：`pip install speech-to-speech[kokoro,faster-whisper]` → CLI 启动 → 四模式（realtime / local / raw-websocket / socket）

**注意**：Qwen3-Omni 不被该仓库直接支持（仅未合并的 draft PR #369），与 §3 结论一致 —— 端到端语音本地化在现有硬件上不做。

---

## 7. 详细实施步骤（分阶段）

### Phase 0 — 环境准备与模型落地（半天）

```bash
# 1. 安装 Ollama（GPU 推理引擎）
curl -fsSL https://ollama.com/install.sh | sh
# 或手动装：下载 linux tar 解压到 /usr/local，见 ollama.com/download/linux

# 2. 拉取主模型（Q4_K_M 默认）
ollama pull qwen3-vl:8b          # 6.1GB，8GB/16GB 通用
# 备选：ollama pull qwen3-vl:4b   # 3.3GB，8GB 更宽裕
# 备选 Gemma：ollama pull gemma4:e2b

# 3. 验证 GPU 推理 + 视觉
ollama run qwen3-vl:8b "描述这张图: /path/to/test.jpg"
# 记录首 token 延迟与 tok/s（预期 8GB 档 30-50 tok/s）

# 4. 验证 OpenAI 兼容 REST
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen3-vl:8b","messages":[{"role":"user","content":"hi"}]}'
```

**验收**：`/v1/chat/completions` 返回 200；nvidia-smi 显示模型常驻 GPU。

### Phase 1 — 纯文本对话本地化（`llm_client.py` / `web_server.py`）

改动最小，仅配置 + 一处代码：

1. `.env`：
   ```env
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_MODEL=qwen3-vl:8b
   # DEEPSEEK_API_KEY 保留或留空（本地不需要）
   ```
2. `english_tutor/llm_client.py` L28-33：放开 api_key 必填校验 —— Ollama 无 key，改为空 key 也允许（或设 `LLM_API_KEY=ollama` 占位，零代码改动）。

**验收**：`python3 run.py` 对话流畅；`web_server.py` 页面问答正常。

### Phase 2 — 摄像头视觉本地化 + 多模态上下文对话（`omni_client.py`）

现有 `_call_cloud_vision` / `_call_cloud_text`（omni_client.py L525-614）已实现 OpenAI 兼容 REST（`image_url` base64 格式 Ollama 原生支持）：

1. `.env`：
   ```env
   LLM_BASE_URL=http://localhost:11434/v1
   OMNI_CLOUD_MODEL=qwen3-vl:8b
   ```
2. `demo.py` L242：`OmniClient(mode=ModelMode.CLOUD)` 已走 cloud 路径 → 实际命中本地 Ollama（base_url 变了）。**零代码改动**，仅去掉 `modalities/audio`、`audio` 请求字段（omni_client.py L548-552 / L290-293）—— 这两个字段 Ollama 不支持会报错，需删除或按 `has_audio` 开关。
3. `vision_manager.py` / `decision_engine.py` 的调用方不动（仍走 `analyze_scene`）。

**核心需求：对话时模型知道上下文（含图片识别内容）** —— qwen3-vl 为 VLM，原生支持"边看图边对话"，两种实现：

- **方式 A（推荐，图片即上下文）**：对话请求从 `_call_text` 改为带图的多模态 chat 请求 —— 复用 `_call_cloud_vision` 的 `content: [image_url(当前帧), text(prompt)]` 格式，把对话历史（现有 `history[-4:]` 逻辑）+ 系统提示拼进 text part，每轮附 1 张当前帧（360×360 小图）。效果与 Gemma 4 综合输入在"知道图片上下文"上等价，视觉理解更强。
- **方式 B（零/极小改动，视觉转述进上下文）**：维持 `analyze_scene`（画面转述为文本描述）+ `_build_dialogue_prompt`（描述+历史拼 prompt）现有链路，qwen3-vl 转述质量高，效果接近方式 A。

**8GB 显存注意**：方式 A 每帧产生数百视觉 token，需控制（每轮 1 张当前帧 + 历史压缩到 `history[-4:]`），8K ctx 够用。

**验收**：`demo.py` 运行，摄像头画面能产出 objects/activity 分析；对话时引用画面内容（如"我看到你在玩红色小汽车"）；单帧分析延迟 < 2s。

### Phase 3 — 实时语音对话本地化（主链路：`agent.py` / `connection.py` + speech-to-speech）

1. 部署 speech-to-speech：
   ```bash
   pip install "speech-to-speech[kokoro,faster-whisper]"
   # 配置 LLM 槽位指向 Ollama（配置文件里 llm 指向 http://localhost:11434/v1/chat/completions）
   speech-to-speech                      # 默认 ws://localhost:8765/v1/realtime
   ```
2. `camera_tutor/agent.py` L101-104 `ws_url`：
   ```python
   # 原：f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...&language=en"
   # 新：f"ws://localhost:8765/v1/realtime?language=en"
   ```
3. `connection.py`：api_key 传空串/dummy（L141 header 可留空 Authorization）；重连/回调逻辑不动。
4. `agent.py` L364-377 的 `session.update` 字段核对：`server_vad`、`threshold` 与 OpenAI Realtime 兼容；阿里云特有字段（如有 `language`/`model` 差异）按 speech-to-speech 实际接受情况微调（需实测）。
5. 视觉输入：speech-to-speech 的 VLM 槽位支持传 `input_image`（Realtime 协议），可与 Phase 2 的 REST 视觉并存，先保持 REST 路径。

**验收**：`python3 camera_tutor/realtime_demo.py --select-devices` 完整跑通 —— 说话→打断→回复全本地；`nvidia-smi` 无网络相关进程；拔网线后功能不降级。

### Phase 4 — TTS / STT 本地化收尾

1. `english_tutor/tts.py` + `web_server.py` L217/552/636：edge-tts 替换为 **Kokoro-82M**（本地）：
   - 方案 a：speech-to-speech 的 TTS 服务直接对外（`--enable_llm_proxy` 或 TTS 端点）；
   - 方案 b：独立 `pip install kokoro` + 本地推理函数，`tts.py` 换实现，接口签名不变（`text → wav 文件路径`）。
2. `english_tutor/stt.py` L74/94：`model_size` 默认 `tiny` → `small`（儿童英语识别率显著提升；`base` 为折中）。模型文件放入 `~/.cache/huggingface/hub`（离线可手动放置，参照 docs/SETUP_RECORD_20260729.md）。
3. edge-tts 保留为可选（`USE_CLOUD_TTS=1` 时仍可用），默认全本地。

**验收**：`/tts` 端点断网可发音；STT 转写英语准确率可接受（可跑 docs/USER_TEST_PLAN.md 的测试集）。

### Phase 5 — 全链路验收与文档

- 用 `docs/USER_TEST_PLAN.md` 的用例过一遍 CLI / Web UI / Camera Tutor 三条链路。
- 记录各链路延迟（首 token、视觉单帧、语音端到端），写入本文件的"实测记录"附录。
- 更新 `SETUP_GUIDE.md` / `DEPLOYMENT_GUIDE.md` 为本地化部署说明。

---

## 8. 代码改造清单（汇总）

| 文件 | 改动 | 工作量 |
|------|------|--------|
| `.env` / `.env.example` | `LLM_BASE_URL`→本地、`LLM_MODEL`→qwen3-vl:8b、新增 `OMNI_LOCAL_*` 开关 | 小 |
| `english_tutor/llm_client.py` | L28-33 api_key 放开为空 | 1 行 |
| `camera_tutor/omni_client.py` | 去掉 `modalities/audio`、`audio` 字段（L290-293、L548-552）；可选：本地模式走 ollama | 小 |
| `camera_tutor/agent.py` | L101-104 `ws_url` → 本地 speech-to-speech；L364 session 字段核对 | 中 |
| `camera_tutor/connection.py` | api_key 兼容空值（L141） | 1 行 |
| `camera_tutor/demo.py` | 无（配置驱动） | 0 |
| `english_tutor/tts.py` + `web_server.py` | edge-tts → Kokoro（接口不变） | 中 |
| `english_tutor/stt.py` | 默认 `tiny` → `small` | 1 行 |
| 新文件：`scripts/local_stack.sh` | 一键起 Ollama（或 vLLM 备选）+ speech-to-speech，引擎只改 base_url | 小 |

> 目标：**不改协议、不改调用方**，全部通过 base_url / 环境变量 / 小函数替换完成，任何时刻可切回云端（见 §10）。

---

## 9. 风险与权衡

| 风险 | 影响 | 缓解 |
|------|------|------|
| qwen3-vl:8b 英语教学对话质量 < 云端 qwen3.5-omni-flash-realtime | 教学体验略降 | 8B 级已够 A1-B2 简单对话；先 PoC 对比，不够再上 16GB 档 Q8_0 或拆 qwen3:8b 对话+视觉分模型 |
| 8GB 档 GPU 内存紧张（6.1GB 权重+KV cache） | OOM / 掉速 | ctx 限 8K、小图 360×360（项目已默认）；备选 qwen3-vl:4b；预留 CPU offload |
| speech-to-speech 与 agent.py session 字段细节不兼容 | 实时链路调试成本 | Phase 3 单独验证；协议同源（OpenAI Realtime），预期小改 |
| STT（faster-whisper small, CPU）实时性 | 语音转写延迟 | 实测；不够用 Parakeet TDT 0.6B（GPU 0.6-1.3GB，8GB 需让位） |
| CUDA 运行时版本（本机 CUDA 13.0 / 驱动 580）与 speech-to-speech 的 `qwentts-cpp-python` wheel（默认 CUDA 12.8）不匹配 | 安装失败 | 换 CPU wheel 或改用 Kokoro（纯 torch/onnx，无此问题） |
| edge-tts 移除后语音质量变化 | 声音不如微软 neural | Kokoro 英语质量已接近；保留 edge-tts 可选开关 |

## 10. 回滚方案

- 所有改动点都有云端/本地切换开关：`.env` 改回 `LLM_BASE_URL` 原值 + `DASHSCOPE_API_KEY` 即恢复云端。
- `agent.py` 的 `ws_url` 加 `OMNI_WS_URL` 环境变量覆盖（默认仍为本地），云端实时链路一行切回。
- TTS 保留 `USE_CLOUD_TTS=1` 回退 edge-tts。
- 代码改动量小（~8 处），git 单分支即可，无需特殊回滚策略。

---

## 11. 待定决策（实施前需确认）

1. **主模型**：`qwen3-vl:8b`（推荐，两档通用）vs `qwen3-vl:4b`（8GB 更稳、质量低一档）。
2. **Gemma 备选是否实测**：8GB 上 `gemma4:e2b` 是否值得花时间验证（权重 7.2GB 余量极小）。
3. **TTS 选型**：Kokoro-82M（推荐，英语教学）vs Qwen3-TTS 1.7B（speech-to-speech 默认，需处理 CUDA wheel 匹配）。
4. **STT 尺寸**：faster-whisper `small`（推荐）vs `base`（更快）。
5. **推理引擎**：8GB 档默认 **Ollama**；16GB 档或生产化是否升级 **vLLM**（AWQ/GPTQ 4bit + `LLM_BASE_URL` 切换）。
