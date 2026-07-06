# 🌊 English Tutor

> **A pretraining environment for your brain's English neural network.**
>
> Not a flashcard app. Not a grammar checker. Built on the insight that adult L2 learners skip the most critical phase — massive passive input — and try to learn language the same way they memorize meeting times.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20Flash-6c5ce7.svg)](https://deepseek.com)

---

## Why This Exists

Modern LLMs train in three phases: **Pretraining** (massive unsupervised input) → **Instruction Tuning** (learning to follow commands) → **LoRA fine-tuning** (small targeted adjustments).

Human language acquisition maps onto the same structure:

```
LLM Pretraining   →  Infant L1: thousands of hours of pure input, no output pressure
Instruction Tuning →  School age: grammar, logic, structured expression
LoRA fine-tuning   →  Adult L2: vocabulary lists, grammar rules on a frozen base
```

**The problem**: almost every English learning tool gives you Phase 3 methods (flashcards, grammar drills) to achieve what requires Phase 1 (massive comprehensible input). You're trying to fine-tune a base model that never saw English pretraining data.

English Tutor is the **missing Phase 1**.

---

## How It Works

### 🧠 The Core Insight

Adult learners forget vocabulary overnight because they store it in **declarative memory** (hippocampus — "what I had for lunch"). Infants don't forget because they store language in **procedural memory** (basal ganglia — "how to ride a bike"). Same input, wrong storage system.

The Immersion Mode activates procedural memory by:

- **Zero translation** — no Chinese anywhere during input
- **Multi-pass exposure** — same content, different sensory channels
- **No output pressure** — you never have to speak, write, or answer questions
- **Parasympathetic state** — guided breathing switches your brain into absorption mode

### 🌊 Immersion Mode — 3-Pass Pretraining

| Pass | Mode | What Happens |
|------|------|-------------|
| **1** 👂 Sound First | Audio only, text hidden | Raw auditory input. Feel rhythm, intonation, emotion. Don't translate. |
| **2** 👁️ Read Along | Text + audio together | Connect written words to sounds. Notice spelling-sound patterns. |
| **3** 🧘 Eyes Closed | Audio only again | Consolidation. Your brain builds direct sound→meaning pathways. |

After Pass 3: key vocabulary revealed (English only, no translations) + a reflective closing thought.

**5 content types** rotate randomly: daily scenes, mini stories, place descriptions, dialogues, first-person observations. **3 difficulty levels**: easy (A2), medium (B1), hard (B2).

All content is generated fresh by DeepSeek — you never run out.

---

## Features

| Feature | Description |
|---------|-------------|
| 🌊 **Immersion** | 3-pass pretraining input with guided breathing and post-session quiet period |
| 💬 **AI Chat** | Natural conversation with Emma, an encouraging tutor at your level |
| 📰 **Reading** | RSS article browsing (Guardian, NPR, NYT, TechCrunch) with difficulty grading |
| 📚 **SR Review** | SM-2 spaced repetition (same algorithm as Anki) for vocabulary retention |
| 🎤 **Voice Input** | Record speech and transcribe with faster-whisper |
| 🔊 **TTS** | Text-to-speech for listening practice (spd-say + edge-tts neural voices) |
| 📊 **Stats** | Learning dashboard: vocabulary, error patterns, streak, SR card progress |
| 📱 **Web UI** | Mobile-first SPA — use on your phone via browser |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [speech-dispatcher](https://freebsoft.org/speechd) (`sudo apt install speech-dispatcher`)
- [DeepSeek API key](https://platform.deepseek.com/)

### Install

```bash
git clone https://github.com/yourusername/english-tutor.git
cd english-tutor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
echo 'DEEPSEEK_API_KEY=sk-your-key-here' > .env
```

### Run — CLI mode

```bash
python3 run.py
```

### Run — Web mode (use on your phone!)

```bash
python3 run.py --web
# Then open http://<server-ip>:8080 on your phone's browser
```

---

## Commands (CLI mode)

| Command | Description |
|---------|-------------|
| *(just type)* | Chat with Emma |
| `/immerse [topic] [difficulty]` | Start immersion session |
| `/read` | Browse today's English articles |
| `/review` | Review due vocabulary cards |
| `/quiz` | Quick vocabulary quiz |
| `/speak` | Read last response aloud |
| `/record [seconds]` | Voice input |
| `/stats` | Learning dashboard |
| `/words` | Vocabulary list |
| `/errors` | Error pattern analysis |
| `/topic` | Suggest a conversation topic |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Architecture

```
run.py
  ├── CLI mode ──► cli.py ──► chat_loop() — REPL with /commands
  └── Web mode ──► web_server.py (FastAPI) ──► mobile SPA

Shared modules:
  llm_client.py          DeepSeek API (streaming + sync, JSON mode)
  immersion_mode.py      3-pass pretraining input engine
  memory.py              Session persistence + vocabulary + error tracking
  tutor_prompt.py        Emma persona + topic suggestions
  reading_mode.py        RSS feed fetcher + article analysis
  review_mode.py         SM-2 review + quiz sessions
  spaced_repetition.py   SM-2 algorithm (Anki-compatible)
  tts.py                 spd-say + edge-tts neural voices
  stt.py                 faster-whisper speech recognition
```

---

## Daily Cost

| Activity | API Cost |
|----------|----------|
| Conversation (30 min) | ¥0.5–2 |
| Immersion (15 min, ~5 passages) | ¥0.05–0.15 |
| Reading (definitions) | ¥0.02–0.10 |
| **Typical daily** | **~¥1–3** |
| **Monthly** | **~¥30–90** |

DeepSeek Flash is ¥1 per million tokens. Immersion content generation is essentially free.

---

## The Design Principles

1. **Pretraining before fine-tuning.** Massive input builds the base model. Grammar and vocabulary come after.
2. **Procedural over declarative.** Activate the basal ganglia, not the hippocampus. Listen, don't memorize.
3. **No translation.** Every translation strengthens the Chinese pathway and weakens the English one.
4. **Consistency over intensity.** 15 minutes daily > 3 hours on Sunday. Sleep consolidation happens every night.
5. **Experience, not achievement.** Don't measure what you learned. Measure what you experienced.

---

## Documentation

- [Full Product Documentation](docs/PRODUCT_DOC.md) — learning theory, architecture, usage guide, cost analysis
- [What Should You Feel After Immersion?](docs/PRODUCT_DOC.md#what-should-you-feel-after-immersion) — the most important section for knowing if you're doing it right

---

## License

MIT

---

*Built on cognitive science. Powered by DeepSeek. Designed for the brain, not for the exam.*
