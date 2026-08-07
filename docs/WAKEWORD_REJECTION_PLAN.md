# 拒识与唤醒词规划（Command Rejection / Wake-word Gating）

> 状态：**方案 A/B 均已实现为独立模块，待集成**（2026-08-07）
> 触发场景：参考车载大模型的"拒识"能力——系统对普通语音不响应，
> 只对可手动配置的特殊指令响应。
>
> 实现：`camera_tutor/command_filter.py`（方案 A 文本门禁）、
> `camera_tutor/voice_gate.py`（A/B 统一门禁，配置文件选模式）。
> 测试：`tests/test_command_filter.py`（27 项）、`tests/test_voice_gate.py`（21 项）。

## 1. 现状

当前本地语音链路（`scripts/local_pipe.py`）对所有声音"照单全收"：

```
麦克风 → VAD(RMS 阈值) → STT(gemma/whisper) → LLM → TTS
```

- 没有任何内容级门禁：VAD 判定为语音的段落都会走完全流程
- 唯一的近似物：`camera_tutor/decision_engine.py` 的 `_is_calling_tutor()`
  （硬编码 "emma/hello/老师" 等呼唤词），但它只服务云端决策引擎的干预判断，
  不在本地实时链路上，且词表写死、不可配置

## 2. 概念区分：两种"门禁"

| | 文本层拒识 | 声学层唤醒（KWS） |
|---|---|---|
| 位置 | STT 之后、LLM 之前 | VAD 之前（音频入口） |
| 判断依据 | 转写文本 vs 指令表 | 声学特征 vs 唤醒词模型 |
| 省什么 | LLM 调用、乱搭话 | 连 STT 都省，算力全省 |
| 指令集 | 任意多条，可随时改 | 通常 1-2 个固定唤醒词，换词要重训 |
| 误伤来源 | STT 转写错误 | 声学相似发音、环境噪音 |

两者不互斥，可叠加：唤醒词进门 + 文本层指令过滤。

## 3. 方案 A：STT 后文本门禁

### 3.1 插入点

```
VAD → STT(文本) → 【拒识门禁】 → LLM → TTS
```

STT 之后是唯一现实的文本门禁位置——拒识判断的是内容，
内容只有转写成文本才可见。

### 3.2 匹配引擎（按成本三级）

1. **精确包含**：归一化（小写、去标点）后逐条子串匹配
2. **模糊匹配**：编辑距离容差 1-2 字符。**儿童场景必需**——
   gemma 把 "look" 听成 "book" 是常事，纯精确匹配的误拒识率会高到不可用
3. **语义匹配**（可选）：指令表 embedding + 余弦相似度。能接住
   "what's this" / "what is this" 等价说法，但引入额外模型调用，
   与"拒识省资源"的初衷部分相悖

### 3.3 状态机

纯车载式是无状态单次判断（每句独立）。但体验上建议加**激活窗口**：

```
[休眠] --命中指令--> [激活(N 秒)] --超时--> [休眠]
```

否则孩子说 "what's this" 系统答了，追问 "why" 又被拒，对话断裂。
窗口时长 N（建议 10s 左右）应可配置。

### 3.4 未命中行为

- **静默**：最"车载"，但孩子会困惑
- **固定提示**：TTS 播预设话术，不经 LLM，成本固定（儿童场景建议此项）
- **计数升级**：连续被拒 N 次后给引导提示

### 3.5 配置设计（复用音频开关的模式）

**已实现**（`camera_tutor/voice_gate.py`）：统一配置文件
`.camera-tutor-data/voice_gate.json`，`mode` 字段选择方案：

```json
{
  "mode": "off",                  // off | text(方案A) | kws(方案B) | kws+text(叠加)
  "text": {                        // 方案 A 配置（command_filter）
    "enabled": true,
    "mode": "command",            // command=整句须命中 | prefix=以前缀开头即放行
    "commands": ["what's this", "sing a song"],
    "wake_prefixes": ["emma"],
    "activation_window_s": 10,
    "on_reject": "canned_reply",  // silent | canned_reply | escalate
    "fuzzy_tolerance": 1
  },
  "kws": {                         // 方案 B 配置（openWakeWord）
    "model_paths": [],            // .onnx 模型路径，空=内置模型
    "threshold": 0.5,
    "activation_window_s": 10,
    "score_cooldown_s": 1.0
  }
}
```

集成后的配套（待做）：`GET/POST /api/voice-gate` + dashboard 编辑区。
指令表建议按导师（persona）维度区分：Grace 面试场景不需要拒识，儿童陪伴需要。

## 4. 方案 B：KWS 唤醒词小模型

### 4.1 原理

```
麦克风流(16kHz) → 20-30ms 帧 → MFCC/log-mel 特征
→ 小型神经网络(DNN/GRU/TCN) 逐帧输出"是唤醒词"概率
→ 平滑 + 阈值 → 触发
```

模型极小（几百 KB~几 MB）、纯 CPU 实时（几个百分点占用）、常开监听。
每个模型带 sensitivity 参数，本质是**误唤醒率 vs 漏唤醒率**的权衡。

### 4.2 方案选型

| 方案 | 定制唤醒词 | 许可 | 结论 |
|------|-----------|------|------|
| Porcupine (Picovoice) | 官网控制台输入词组，云端代训练 | 商用收费 | 最省事 |
| **openWakeWord** | TTS 合成数千条带噪假数据自动训练，无需真人录音 | Apache 2.0 | **推荐**（Home Assistant 同款） |
| Mycroft Precise | 自录数百条样本 | 停更 | 不推荐 |
| Snowboy | — | 2020 关停 | 不考虑 |

命令级拒识的声学方案还有 Speech-to-Intent（如 Picovoice Rhino）：
给定有限命令语法，语法外输入直接"未理解"。识别率高、天然拒识，
但语法需穷举，做不了自由对话。

### 4.3 接入架构

```
麦克风 ──→ KWS 小模型(常开)
              │ 检出唤醒词
              ▼ 开门 N 秒
         VAD → STT → LLM → TTS   （现有链路，零改动）
              │
              ▼ 超时未说话
            关门
```

实施路径（真到做的时候）：合成数据 → 官方 notebook 训练（约 1 小时）
→ `local_pipe.py` 音频入口加约 20 行门控类。

## 5. 关键权衡与风险

- **文本门禁省 LLM，声学门禁全省**；但 KWS 解决不了"任意指令集拒识"，
  指令集过滤仍需文本层（或 Rhino 式有限语法）
- **误拒识率绑死在 STT 准确率上**：E4B 转写儿童语音本有误差，
  指令越短、越口语化，模糊匹配的误触发与精确匹配的误拒识越难平衡
- 先验数据再定阈值，不要先上线再调

## 6. 分期规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | **数据评估**：回放 `logs/local_pipe.log` 历史转写，模拟不同指令表/匹配策略下的拒识率与误拒识率 | 待做 |
| P1 | 文本门禁（command_filter + voice_gate text 模式） | **已完成**（2026-08-07）：模块 + local_pipe 集成（STT 后 check_text，拒识话术直接 TTS）+ `/api/voice-gate` + dashboard Device 页编辑区；配置 mtime 热重载，默认 off 对现有对话零影响 |
| P2 | KWS 唤醒词门禁（voice_gate kws 模式，openWakeWord 后端） | **框架已接入**：local_pipe 音频入口 feed_audio 门控（未唤醒丢音频，异常时 fail-open 不影响对话）；待做：`pip install openwakeword` + 训练/获取唤醒词模型 |
| P3 | （可选）有限语法命令层：Rhino 或等价方案 | 未开始 |

P0 是纯分析工作，不碰线上链路，随时可做。
