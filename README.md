# 🌊 English Tutor

> **A pretraining environment for your brain's English neural network.**
>
> Not a flashcard app. Not a grammar checker. Built on the insight that adult L2 learners skip the most critical phase — massive passive input — and try to learn language the same way they memorize meeting times.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20Flash-6c5ce7.svg)](https://deepseek.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How It Works](#how-it-works)
- [Example Session](#example-session)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
- [Architecture](#architecture)
- [Web UI (Mobile)](#web-ui-mobile)
- [Tech Stack](#tech-stack)
- [Daily Cost](#daily-cost)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

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

### The Core Insight

Adult learners forget vocabulary overnight because they store it in **declarative memory** (hippocampus — "what I had for lunch"). Infants don't forget because they store language in **procedural memory** (basal ganglia — "how to ride a bike"). Same input, wrong storage system.

The Immersion Mode activates procedural memory by:

- **Zero translation** — no Chinese anywhere during input. Translation strengthens the wrong pathway.
- **Multi-pass exposure** — same content through different sensory channels builds robust traces.
- **No output pressure** — you never have to speak, write, or answer questions. Pure absorption.
- **Parasympathetic state** — 30-second guided breathing (4s in → 2s hold → 6s out) activates the vagus nerve, switching your brain from "fight/flight" to "rest/digest" — the only state where procedural learning happens.
- **Sleep-aligned** — optimal timing (before sleep or after waking) exploits REM consolidation windows.

### Immersion Mode — 3-Pass Pretraining

| Pass | Mode | What Happens |
|------|------|-------------|
| **1** 👂 Sound First | Audio only, text hidden | Raw auditory input. Feel rhythm, intonation, emotion. Don't translate. |
| **2** 👁️ Read Along | Text + audio together | Connect written words to sounds. Notice spelling-sound patterns. |
| **3** 🧘 Eyes Closed | Audio only again | Consolidation. Brain builds direct sound→meaning pathways. |

After Pass 3: key vocabulary revealed (English only, no translations) + a reflective closing thought.

**5 content types**: daily scenes 🌅, mini stories 📖, place descriptions 🏞️, dialogues 💬, observations 🔍.  
**3 difficulty levels**: easy (A2), medium (B1), hard (B2).  
All content is generated fresh by DeepSeek — unlimited, never repeats.

### What You Should Feel Afterwards

Not "I learned 3 words." That's declarative memory. The real signs of procedural learning are:

- A vague, fuzzy impression — you couldn't retell it, but a hazy outline lingers
- A word or phrase echoing in your head on its own (no conscious recall)
- Mild, contented fatigue — neurons consume ATP to form new synapses
- An urge to stay quiet — your brain is running background consolidation

If you feel these, you're doing it right. [Full guide →](docs/PRODUCT_DOC.md#what-should-you-feel-after-immersion)

---

## Example Session

Here's what a typical immersion session looks like in CLI mode:

```
$ python3 run.py

   _____ _ _   _        _____    _              _
  | ____(_) | | |_   _ |_   _|__| |_ _ __ ___ (_)_ __
  |  _| |_| |_| | | | | | |/ _ \ __| '_ ` _ \| | '_ \
  | |___| |  _  | |_| | | |  __/ |_| | | | | | | | | |
  |_____|_|_| |_|\__,_| |_|\___|\__|_| |_| |_|_|_| |_|

Your AI English Tutor — practice daily, improve naturally

Type anything to start chatting with Emma, your English tutor.
Type /help to see available commands.

You: /immerse

┌─────────────────────────────────────────────────────────────┐
│ 🌊  English Immersion Mode                                 │
│ Phase-1 Pretraining Input                                  │
│                                                             │
│ "You can't learn to swim by reading about water."          │
│ You have to get in. This is your pool.                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🧘  Before We Begin — 3 Deep Breaths                       │
│                                                             │
│ This activates your parasympathetic nervous system:        │
│ the state where real language absorption happens.          │
│ Not memorization. Not analysis. Pure absorption.           │
└─────────────────────────────────────────────────────────────┘

Breath 1/3
  ↗ In  4...3...2...1
  Hold...
  ↘ Out 6...5...4...3...2...1

Breath 2/3
  ...

✓ Ready. Your brain is now in learning mode.

✨ Generating your first immersion passage...

┌─────────────────────────────────────────────────────────────┐
│ Pass 1/3 — 👂 Sound First                                  │
│ Listen only. Don't try to understand every word.           │
│ Let the rhythm and intonation wash over you.               │
│                                                             │
│ 💡 Relax your shoulders. Unclench your jaw.                │
│    You're not being tested. You're just listening.         │
└─────────────────────────────────────────────────────────────┘
🎧 Now playing...
✓ Done — What did you feel? Any words stand out?

Press Enter for Pass 2
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🌊 **Immersion** | 3-pass pretraining input with guided breathing and post-session quiet period |
| 💬 **AI Chat** | Natural conversation with Emma, an encouraging tutor who adapts to your level |
| 📰 **Reading** | RSS article browsing (Guardian, NPR, NYT, TechCrunch) with difficulty grading, one-tap definitions, and summaries |
| 📚 **SR Review** | SM-2 spaced repetition (same algorithm as Anki) — cards added automatically from conversations |
| 🎤 **Voice Input** | Record speech via microphone and transcribe with faster-whisper |
| 🔊 **TTS** | Text-to-speech — quick `spd-say` for CLI, neural `edge-tts` voices for web |
| 📊 **Stats** | Dashboard: total sessions, messages, vocabulary, error patterns with frequency bars, SR card progress, daily streak |
| 📱 **Web UI** | Mobile-first SPA — Chat, Immerse, and Stats tabs. Use on your phone via browser |

---

## Quick Start

### Prerequisites

| Dependency | Purpose | Install |
|-----------|---------|---------|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| speech-dispatcher | CLI TTS | `sudo apt install speech-dispatcher` |
| DeepSeek API key | LLM | [platform.deepseek.com](https://platform.deepseek.com) |

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
python3 run.py --web --host 0.0.0.0 --port 8080
# Then open http://<server-ip>:8080 on your phone's browser
```

---

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | *(required)* | DeepSeek API key |
| `LLM_MODEL` | `deepseek-v4-flash` | Model name (any OpenAI-compatible) |
| `LLM_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `ENGLISH_TUTOR_DATA_DIR` | `.english-tutor-data/` | Session, stats, and SR card storage |

Example `.env`:

```env
DEEPSEEK_API_KEY=sk-abc123...
# Optional:
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
ENGLISH_TUTOR_DATA_DIR=.english-tutor-data
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| *(just type)* | Chat with Emma — she corrects errors gently and introduces new vocabulary |
| `/immerse [topic] [difficulty]` | Start immersion session (e.g. `/immerse cooking easy`) |
| `/read` | Browse today's English articles from major news sources |
| `/review` | Review due spaced-repetition vocabulary cards |
| `/quiz` | Quick vocabulary quiz — test your retention |
| `/speak` | Read Emma's last response aloud via TTS |
| `/record [seconds]` | Record voice input and transcribe (e.g. `/record 5`) |
| `/stats` | Learning dashboard with progress bars |
| `/words` | Show your vocabulary list |
| `/errors` | Error pattern analysis with frequency bars and practice tips |
| `/topic` | Get a conversation topic suggestion |
| `/new` | Start a fresh conversation session |
| `/help` | Show all available commands |
| `/quit` | Save and exit |

---

## Architecture

```
run.py
  ├── CLI mode ──► cli.py ──► chat_loop() — Rich-powered REPL with /commands
  └── Web mode ──► web_server.py (FastAPI + uvicorn) ──► mobile SPA

Shared modules:
  llm_client.py          DeepSeek API client
                         ├── chat()        SSE streaming
                         ├── chat_sync()   Non-streaming + response_format
                         └── httpx.AsyncClient with Bearer auth

  immersion_mode.py      3-pass pretraining input engine
                         ├── ImmersionSession     State machine
                         ├── ImmersionContent     Dataclass
                         ├── _breathing_guide()   30s parasympathetic activator
                         ├── _clean_passage()     Fallback text sanitizer
                         └── _parse_json_response()  Robust JSON extractor

  memory.py              JSON-based persistence
                         ├── ConversationMemory   Sessions, vocabulary, errors
                         ├── Auto-save on every message
                         └── Streak calculation

  spaced_repetition.py   SM-2 algorithm (Anki-compatible)
                         ├── Card dataclass with full SM-2 state
                         ├── get_due_cards() / get_stats()
                         └── Persisted to spaced_repetition.json

  tutor_prompt.py        Emma persona definition + topic suggestions

  reading_mode.py        RSS reader + article browser
                         ├── FeedFetcher (concurrent RSS + readability extraction)
                         ├── Difficulty grading (word + sentence length heuristics)
                         └── CET-4 word filtering

  review_mode.py         /review and /quiz interactive sessions

  tts.py                 Two-tier TTS
                         ├── spd-say (quick, CLI)
                         └── edge-tts (neural voices, web)

  stt.py                 Speech-to-text via arecord + faster-whisper

  web_server.py          10 REST endpoints + SSE streaming
  static/index.html      Mobile-first SPA (3 tabs, dark theme)
```

---

## Web UI (Mobile)

Start with `python3 run.py --web`, then open on your phone:

```
┌─────────────────────────────────┐
│     🌊 English Tutor            │
│     Phase-1 Pretraining Input   │
├─────────────────────────────────┤
│                                 │
│  ┌─ Chat View ────────────────┐ │
│  │ Emma: Hi! I'm Emma, your   │ │
│  │ English tutor. 😊          │ │
│  │                            │ │
│  │ You: I want to practice    │ │
│  │ listening today.           │ │
│  │                            │ │
│  │ Emma: Great idea! ...      │ │
│  │                            │ │
│  │ [________________] [Send]  │ │
│  └────────────────────────────┘ │
│                                 │
│  ┌─ Immerse View ─────────────┐ │
│  │ [B1▼] [🎲 Random▼] [topic]│ │
│  │ [✨ Generate Passage]      │ │
│  │                            │ │
│  │  ●  ○  ○    Pass 1/3      │ │
│  │  👂 Sound First            │ │
│  │  Listen only...            │ │
│  │  [▶️ Audio Player        ] │ │
│  │  [◀ Back]  [Next ▶]       │ │
│  └────────────────────────────┘ │
│                                 │
│  ┌─ Stats View ───────────────┐ │
│  │ 📊 42 sessions · 350 msgs  │ │
│  │ 📚 68 words · 12 mastered  │ │
│  │ 📝 Errors: tense 8x ████   │ │
│  └────────────────────────────┘ │
│                                 │
├─────────────────────────────────┤
│  [💬 Chat] [🌊 Immerse] [📊 Stats] │
└─────────────────────────────────┘
```

### Web API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the SPA |
| `POST` | `/api/chat/stream` | SSE streaming chat completion |
| `POST` | `/api/immerse/generate` | Generate immersion content (JSON) |
| `POST` | `/api/speak` | Generate TTS MP3 via edge-tts |
| `GET` | `/api/audio/{filename}` | Serve generated MP3 |
| `GET` | `/api/stats` | Learning statistics |
| `GET` | `/api/vocabulary` | Vocabulary list |
| `GET` | `/api/errors` | Error pattern analysis |
| `GET` | `/api/review/due` | Due SR cards |
| `POST` | `/api/review/submit` | Submit review result |
| `GET` | `/api/topic` | Conversation topic suggestion |
| `GET` | `/api/health` | Server health check |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **LLM** | DeepSeek Flash (OpenAI-compatible API) |
| **CLI** | Rich (panels, markdown, tables, prompts) |
| **Web** | FastAPI + uvicorn + vanilla JS SPA |
| **HTTP** | httpx (async) |
| **TTS** | speech-dispatcher (spd-say) + edge-tts (Microsoft neural) |
| **STT** | faster-whisper |
| **RSS** | feedparser + readability-lxml |
| **Audio** | arecord (Linux ALSA) |

### Python Dependencies

```
rich>=13.0.0          # Terminal UI
httpx>=0.27.0         # Async HTTP
fastapi>=0.100.0      # Web framework
uvicorn>=0.23.0       # ASGI server
pydantic>=2.0.0       # Data validation
python-dotenv>=1.0.0  # Environment config
feedparser>=6.0.0     # RSS parsing
readability-lxml>=0.8.0  # Article extraction
lxml>=5.0.0           # XML/HTML parsing
edge-tts>=6.0.0       # Neural TTS
faster-whisper>=1.1.0 # Speech recognition
```

---

## Daily Cost

| Activity | Est. API Cost |
|----------|--------------|
| Conversation (30 min) | ¥0.5–2 |
| Immersion (15 min, ~5 passages) | ¥0.05–0.15 |
| Reading (article definitions) | ¥0.02–0.10 |
| **Typical daily** | **~¥1–3** |
| **Monthly** | **~¥30–90** |

DeepSeek Flash: ¥1/million tokens. Immersion passages are ~500 tokens each = ¥0.0005/passage. The most valuable feature costs almost nothing.

---

## Design Principles

1. **Pretraining before fine-tuning.** Massive input builds the base model. Grammar and vocabulary come after.
2. **Procedural over declarative.** Activate the basal ganglia, not the hippocampus. Experience, don't memorize.
3. **No translation.** Every translation strengthens the L1 pathway and weakens the L2 one. Zero Chinese during input.
4. **Consistency over intensity.** 15 minutes daily > 3 hours on Sunday. Sleep consolidation happens every night.
5. **Experience, not achievement.** Don't measure "what did I learn?" Measure "what did I experience?"

---

## Roadmap

- [x] CLI chat with streaming LLM responses
- [x] Immersion mode — 3-pass pretraining input
- [x] Guided breathing (parasympathetic activation)
- [x] RSS reading mode with difficulty grading
- [x] SM-2 spaced repetition
- [x] Voice input (STT) and TTS output
- [x] Learning statistics and error tracking
- [x] Web UI with mobile-first SPA
- [ ] Session history browser in web UI
- [ ] Immersion streaks and daily reminders
- [ ] Adaptive difficulty based on user performance
- [ ] Offline content pack (pre-generated, no API needed)
- [ ] Docker image for one-command deployment
- [ ] PWA support (install to phone home screen)

---

## Troubleshooting

### "DEEPSEEK_API_KEY not found"

Create a `.env` file in the project root:
```bash
echo 'DEEPSEEK_API_KEY=sk-your-key-here' > .env
```

### "spd-say: command not found" (CLI TTS not working)

Install speech-dispatcher:
```bash
sudo apt install speech-dispatcher
```

### No audio on web UI

The web UI uses edge-tts for TTS. Ensure `edge-tts` is installed:
```bash
pip install edge-tts
```
If audio still doesn't play, check that your phone's silent mode is off and volume is up.

### "arecord: command not found" (voice recording not working)

Install ALSA utilities:
```bash
sudo apt install alsa-utils
```

### Web server can't be reached from phone

1. Ensure `--host 0.0.0.0` is set (default)
2. Check your firewall: `sudo ufw allow 8080`
3. Verify you're on the same network
4. Try: `python3 run.py --web --host 0.0.0.0 --port 8080`

### Immersion content shows "Warning: Could not parse JSON"

This means the LLM didn't return valid JSON. The system has a multi-layer fallback that cleans the raw text — content should still display correctly. If it happens repeatedly, the `response_format: json_object` parameter (enabled by default) should eliminate this. If not, check your `LLM_MODEL` setting.

---

## Documentation

- [Full Product Documentation](docs/PRODUCT_DOC.md) — learning theory, architecture, usage guide, cost analysis
- [What Should You Feel After Immersion?](docs/PRODUCT_DOC.md#what-should-you-feel-after-immersion) — how to know if you're doing it right

---

## License

MIT

---

*Built on cognitive science. Powered by DeepSeek. Designed for the brain, not for the exam.*
