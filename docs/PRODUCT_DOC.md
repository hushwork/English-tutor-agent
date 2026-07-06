# English Tutor — Product Documentation

> **A CLI-based English learning assistant built on cognitive science and LLM training principles.**
>
> Not a flashcard app. Not a grammar checker. A **pretraining environment** for your brain's English neural network.

---

## Table of Contents

1. [Why This Exists](#why-this-exists)
2. [The LLM Analogy: Three-Stage Learning Framework](#the-llm-analogy-three-stage-learning-framework)
3. [Why Adults Forget and Infants Don't](#why-adults-forget-and-infants-dont)
4. [Architecture](#architecture)
5. [Features](#features)
6. [Commands](#commands)
7. [Immersion Mode](#immersion-mode)
8. [How to Use for Maximum Effect](#how-to-use-for-maximum-effect)
9. [Setup](#setup)
10. [Daily Cost](#daily-cost)

---

## Why This Exists

Most English learning tools assume you learn by:

1. Memorizing vocabulary (flashcards, word lists)
2. Studying grammar rules
3. Practicing output (speaking, writing)

These are all **Phase 2/3 methods** — instruction tuning and fine-tuning. But they skip the most critical phase that every infant goes through: **massive passive input without translation**.

The result: you can pass English exams but freeze in real conversation. You know the word but can't hear it at native speed. You translate in your head before speaking.

This tool is designed to fix that gap.

---

## The LLM Analogy: Three-Stage Learning Framework

Modern large language models go through three distinct training stages. Human language acquisition maps almost perfectly onto this same structure:

```
LLM Training Pipeline              Human Language Development
──────────────────────────────     ─────────────────────────────
Phase 1: Pretraining              0–6 years: L1 Foundation
  Trillions of tokens               Thousands of hours of input
  Unsupervised learning             Unconscious statistical learning
  Goal: next-token prediction       Goal: sound → meaning mapping
  Output: Foundation Model          Output: Native neural base network

Phase 2: Instruction Tuning        6–18 years: Logic / Grammar
  Supervised fine-tuning            School education
  Learn to follow instructions      Learn to organize logic, argue
  Output: Instruct Model            Output: Structured language ability

Phase 3: LoRA / QLoRA              18+ years: Adult L2 Learning
  Base weights FROZEN               L1 base network mostly frozen
  Only update low-rank adapters     Can only add "peripheral tuning"
  Small data + strong constraints   Memorize words, learn grammar rules
```

### The Critical Insight

When adults try to learn English, they almost exclusively use **Phase 3 methods** (vocabulary lists, grammar rules = LoRA fine-tuning on a few hundred data points) to achieve what requires **Phase 1** (massive comprehensible input = pretraining on millions of data points).

This is like trying to teach LLaMA Japanese by only giving it 50 translated sentence pairs with LoRA. It can learn to parrot "konnichiwa", but it will never develop genuine Japanese intuition — because **Japanese was never in its pretraining corpus**.

### What This Means for You

| Your Goal | What You Need | What You're Doing |
|-----------|--------------|-------------------|
| Understand native speech | Phase 1: massive audio input | Phase 3: studying phonetics rules |
| Think in English | Phase 1: direct concept binding | Phase 3: translation exercises |
| Speak fluently | Phase 1: implicit pattern learning | Phase 3: grammar drills |

The Immersion Mode in this tool is designed to provide **Phase 1 pretraining input** — the missing piece.

---

## Why Adults Forget and Infants Don't

### Two Memory Systems

| | Declarative Memory (Adults) | Procedural Memory (Infants) |
|---|---|---|
| **Brain region** | Hippocampus | Basal ganglia, cerebellum |
| **Characteristic** | Fragile, requires conscious recall | Robust, automatic activation |
| **Analogy** | Forgetting what you had for lunch | Never forgetting how to ride a bike |
| **After sleep** | Lost if not consolidated (Ebbinghaus curve) | Automatically strengthened during sleep |

When you memorize `apple = 苹果`, you're using the **same neural pathway** you use to remember "meeting at 3 PM Wednesday". That pathway is designed for short-term storage and rapid forgetting — evolutionarily, you don't need to remember every trivial detail.

An infant's hippocampus isn't fully developed yet. They can only use the procedural memory system, which is **"learn once, never forget"** type — like muscle memory.

### Sleep Consolidation Gap

| | Infant | Adult (L2 learner) |
|---|---|---|
| Daily sleep | 14–17 hours | 7–8 hours |
| REM sleep % | ~50% | ~20% |
| English input / day | ~10 hours | ~30 minutes |
| **Effective consolidation time** | **~5 hours** | **~6 minutes** |

That's a **50× gap** in how much consolidation happens each night.

### Interference: The Chinese Network Always Wins

Your brain has a super-efficient Chinese sound→meaning network built over decades. When English sound enters:

```
English sound input
    ↓
  ┌→ Chinese "苹果" network (STRONG, instantly activated) ← what you actually use
  │
  └→ English direct concept network (WEAK, barely activated) ← what you should build
```

During sleep consolidation, the brain **prunes weak synapses and keeps strong ones**. The Chinese network wins every time. The weak English traces are treated as noise and deleted overnight.

The infant has no competitor. Their `apple` mapping is the only one. Sleep consolidation is clean and one-directional.

### The Solution

1. **Block the translation path** — zero Chinese during input sessions
2. **Multi-sensory binding** — audio + text + emotional context simultaneously
3. **Sleep-spaced repetition** — a little every day, let sleep do the consolidation
4. **Relaxed state** — procedural memory only activates under parasympathetic (rest/digest) nervous system

---

## Architecture

```
run.py
  └─ cli.main()
       ├─ LLMClient ──► httpx.AsyncClient ──► POST /chat/completions (SSE streaming)
       ├─ ConversationMemory ──► .english-tutor-data/session_*.json + stats.json
       ├─ SpacedRepetition ──► .english-tutor-data/spaced_repetition.json (SM-2)
       │
       └─ chat_loop()  ← the REPL
            ├─ /chat    → Default: streaming conversation with Emma
            ├─ /read    → reading_mode.py → RSS feed reader + article analysis
            ├─ /immerse → immersion_mode.py → 3-pass pretraining input
            ├─ /review  → review_mode.py → SpacedRepetition.get_due_cards()
            ├─ /quiz    → review_mode.py → random vocabulary quiz
            ├─ /speak   → tts.py speak_now() → spd-say
            ├─ /record  → stt.py record_and_transcribe() → arecord + faster-whisper
            ├─ /stats   → memory.get_stats_summary(sr_stats)
            ├─ /words   → memory.get_vocabulary()
            ├─ /errors  → memory.stats["common_errors"]
            └─ /topic   → tutor_prompt.build_topic_suggestion()
```

### Module Map

| Module | Role | Training Phase |
|--------|------|---------------|
| `immersion_mode.py` | Massive passive input (listening + reading) | **Phase 1: Pretraining** |
| `reading_mode.py` | RSS article reading + vocabulary extraction | Phase 1 + 2 |
| `tutor_prompt.py` + `cli.py` | Conversation practice with Emma | Phase 2: Instruction |
| `review_mode.py` + `spaced_repetition.py` | SM-2 vocabulary review | Phase 3: Fine-tuning |
| `tts.py` | Text-to-speech (spd-say + edge-tts) | Cross-cutting |
| `stt.py` | Speech-to-text (faster-whisper) | Cross-cutting |
| `memory.py` | Session persistence + stats + error tracking | Infrastructure |

---

## Features

### 1. AI Conversation (`/chat` — default mode)

Chat with Emma, a friendly English tutor persona. She:
- Speaks only English (unless you explicitly ask for Chinese help)
- Gently corrects significant errors AFTER you finish your thought
- Adapts vocabulary to your level (CET-4 / B1 by default)
- Introduces new words naturally in context
- Tracks error patterns for later review

### 2. Immersion Mode (`/immerse`)

**The centerpiece.** 3-pass pretraining input designed to build direct sound→meaning neural pathways. See [Immersion Mode](#immersion-mode) below for full details.

### 3. Reading Mode (`/read`)

Fetches today's articles from Guardian, NPR, NYT, and TechCrunch via RSS. Features:
- Article browsing with difficulty grading
- One-tap word definition via LLM
- Article-level vocabulary extraction
- Article summaries

### 4. Spaced Repetition (`/review`, `/quiz`)

SM-2 algorithm (same as Anki) for vocabulary retention:
- Cards added from conversation and reading modes
- Review due cards with quality scoring (0-5)
- Quiz mode for quick testing
- Stats: total, learned (21d+), learning, due

### 5. Voice I/O (`/speak`, `/record`)

- `/speak`: Read the last response aloud (spd-say TTS)
- `/record [N]`: Record N seconds of speech → transcribe → use as input

### 6. Progress Tracking (`/stats`, `/words`, `/errors`)

- `/stats`: Dashboard with sessions, messages, streak, vocabulary, SR cards
- `/words`: Last 20 saved vocabulary entries
- `/errors`: Error pattern analysis with frequency bars and practice tips

---

## Commands

| Command | Description |
|---------|-------------|
| *(just type)* | Chat with Emma |
| `/help` | Show all commands |
| `/new` | Start a new conversation |
| `/immerse [topic] [difficulty]` | **Immersion mode** — pretraining input |
| `/read` | Browse today's English articles |
| `/speak` | Read last response aloud |
| `/record [seconds]` | Voice input (e.g. `/record 5`) |
| `/review` | Review due vocabulary cards |
| `/quiz` | Quick vocabulary quiz |
| `/topic` | Suggest a conversation topic |
| `/stats` | Learning statistics dashboard |
| `/words` | Vocabulary list |
| `/errors` | Error pattern analysis |
| `/save` | Save session |
| `/quit` | Exit |

---

## Immersion Mode

### The 3-Pass Protocol

Each piece of content goes through three passes:

| Pass | Name | Mode | Purpose |
|------|------|------|---------|
| **1** | 👂 Sound First | Audio only, no text displayed | Raw auditory input. Feel rhythm and intonation. Notice emotion. |
| **2** | 👁️ Read Along | Text + audio together | Connect written words to sounds. Notice spelling-sound mismatches. |
| **3** | 🧘 Eyes Closed | Audio only again | Consolidation. Your brain now has the text memory — can it map directly? |

After Pass 3: key vocabulary is revealed (English only, no translations) and a reflective closing thought is shown.

### Content Types (randomized)

| Type | Example |
|------|---------|
| 🌅 Daily Scene | A vivid moment from everyday life |
| 📖 Mini Story | A short narrative with an emotional arc |
| 🏞️ Place | Rich sensory description of a location |
| 💬 Dialogue | Natural conversation between two people |
| 🔍 Observation | First-person reflection on something noticed |

### Difficulty Levels

| Level | CEFR | Description |
|-------|------|-------------|
| `easy` | A2 | Simple sentences, high-frequency words, slower pace |
| `medium` | B1 | Natural CET-4 level (default) |
| `hard` | B2 | Complex structures, richer vocabulary, abstract ideas |

### Immersion Commands

| Key | Action |
|-----|--------|
| `Enter` / `n` | Next → generate new content |
| `a` | Again — replay current content from Pass 1 |
| `w` | Show key vocabulary |
| `d` | Change difficulty |
| `t` | Set a specific topic |
| `-` / `+` | Adjust speech speed |
| `s` | Skip this content |
| `?` | Show help |
| `q` | Quit immersion mode |

### Design Principles

1. **Zero Chinese output** — the LLM prompt forbids any Chinese. All content is pure English.
2. **Sensory-rich generation** — content focuses on what you SEE, HEAR, SMELL, FEEL, TASTE. Concrete over abstract.
3. **Natural repetition** — 3-5 key words repeat 2-3 times in context, not in a vocabulary list.
4. **Emotional engagement** — content aims for surprise, wonder, nostalgia, curiosity. Emotion is memory glue.
5. **No output pressure** — you never have to speak, write, or answer questions. Pure input.

---

## How to Use for Maximum Effect

### The Optimal Daily Routine

```
1. Sit in your fixed study spot              ← context-dependent learning
2. Put on headphones                         ← direct auditory pathway
3. Turn phone screen-down, close all tabs    ← single English channel
4. Take 3 deep breaths (4s in, 6s out)       ← activate parasympathetic system
5. /immerse medium                           ← start immersion
6. Complete 2-3 passages (~15 minutes)        ← don't overdo it
7. Sit quietly for 2 minutes after           ← let the brain settle
8. Ideally: do this right before sleep        ← immediate sleep consolidation
```

### Rules of Thumb

| Do | Don't |
|----|-------|
| Listen for feeling, not every word | Try to translate in your head |
| Let meaning emerge naturally | Ask "what does this word mean?" during listening |
| 15-20 min daily, every day | 3 hours once a week |
| Do it before sleep or right after waking | Do it while multitasking |
| Use headphones | Use speakers |
| Same place, same time | Random locations and times |

### Why These Rules Work

- **Frequency > Intensity**: Synaptic plasticity requires metabolic resources. 20 minutes is the sweet spot before your brain shifts from "absorbing" to "coping". But sleep consolidation happens EVERY night — daily sessions exploit this.
- **Before sleep**: REM sleep immediately consolidates procedural memories. Input → sleep → consolidation. Maximum retention.
- **After waking**: Your prefrontal cortex (analysis system) isn't fully online yet, but basal ganglia (procedural system) is active. This is the closest adult state to an infant brain.
- **Fixed location**: Context-dependent memory — the environment becomes a retrieval cue. Sit in the same chair, your brain automatically enters "English input mode".
- **Headphones**: Creates an "intracranial sound field" that bypasses some auditory filtering. Closer to how you hear your own inner voice.
- **No multitasking**: Dual-task scenarios force the brain to use hippocampus (declarative). Procedural learning requires single-focus, relaxed attention.
- **No translation**: Every time you translate, you strengthen the Chinese pathway and weaken the English one. The goal is to let the English pathway grow strong enough to compete.

### When NOT to do immersion

- Right after a heavy meal (blood in stomach, brain in energy-save mode)
- When anxious or facing a deadline (cortisol blocks synaptic plasticity)
- Right after scrolling Chinese social media (Chinese network is primed, will dominate)
- When exhausted (tired brain consolidates strong connections, prunes weak ones)

---

## Setup

### Prerequisites

- Python 3.10+
- `spd-say` (for TTS) — install with `sudo apt install speech-dispatcher`
- DeepSeek API key

### Installation

```bash
cd ~/workspace/english-tutor
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
DEEPSEEK_API_KEY=sk-your-key-here
# Optional:
# LLM_MODEL=deepseek-v4-flash
# LLM_BASE_URL=https://api.deepseek.com
# ENGLISH_TUTOR_DATA_DIR=.english-tutor-data
```

### Running

```bash
python3 run.py
```

---

## Daily Cost

| Activity | Est. API Cost |
|----------|--------------|
| Conversation (30 min) | ¥0.5–2 |
| Immersion (15 min, ~5 passages) | ¥0.05–0.15 |
| Reading (article definitions) | ¥0.02–0.10 |
| **Daily total (typical use)** | **~¥1–3** |
| **Monthly** | **~¥30–90** |

DeepSeek Flash is ¥1 per million tokens — immersion content generation is extremely cheap (each passage is ~300 tokens prompt + ~200 tokens output = ~500 tokens = ¥0.0005 per passage).

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│  PRETRAINING (Immersion Mode)                               │
│  Daily 15 min → 90 hours/year of pure English input         │
│  This is the 99% most learners skip.                        │
├─────────────────────────────────────────────────────────────┤
│  INSTRUCTION TUNING (Chat with Emma)                        │
│  Active conversation → grammar, output practice             │
│  This is the 1% most learners focus ALL their energy on.    │
├─────────────────────────────────────────────────────────────┤
│  LoRA FINE-TUNING (Spaced Repetition)                       │
│  Targeted vocabulary retention → fill specific gaps         │
│  Useful but ONLY after the base model has enough input.     │
└─────────────────────────────────────────────────────────────┘
```

**Most English learning tools give you Phase 3 and call it a day.**

**This one starts at Phase 1 — because that's where language actually begins.**

---

*Built on cognitive science. Powered by DeepSeek. Designed for the brain, not for the exam.*
