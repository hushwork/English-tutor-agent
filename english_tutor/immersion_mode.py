"""Immersion Mode — Phase-1 pretraining style language input.

Based on the insight that adult L2 learners skip the critical
"pretraining" phase that infants go through: massive passive input
without translation. This mode provides multi-pass, multi-sensory
English input designed to build direct sound→meaning neural pathways.

Analogy:
  LLM Pretraining  →  Infant L1 acquisition (massive input, no task)
  Instruction Tuning → School-age logic/grammar training
  LoRA fine-tuning   → Adult L2 learning (frozen base, small adapter)

Most adult L2 learners try to use "LoRA" (grammar rules, vocabulary lists)
to achieve what requires "pretraining" (massive comprehensible input).
This module provides the missing pretraining phase.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from english_tutor.llm_client import LLMClient
from english_tutor.tts import speak_now

console = Console()
WIDTH = 78

# ── Content types for variety ──────────────────────────────────────

CONTENT_TYPES = [
    "daily_scene",
    "mini_story",
    "place_description",
    "dialogue",
    "observation",
]

CONTENT_TYPE_LABELS: dict[str, str] = {
    "daily_scene": "🌅 Daily Scene",
    "mini_story": "📖 Mini Story",
    "place_description": "🏞️ Place",
    "dialogue": "💬 Dialogue",
    "observation": "🔍 Observation",
}

DIFFICULTY_LABELS: dict[str, str] = {
    "easy": "A2 — Easy",
    "medium": "B1 — Medium",
    "hard": "B2 — Challenging",
}

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "easy": "Simple sentences, high-frequency words, slower conceptual pace. Like talking to a child.",
    "medium": "Natural B1/CET-4 level, some connecting words, everyday vocabulary with occasional richer expressions.",
    "hard": "More complex sentence structures, richer vocabulary, abstract ideas, idiomatic expressions.",
}

# ── Topic seeds (no Chinese, varied domains) ───────────────────────

TOPIC_SEEDS = [
    "a quiet moment in a busy day",
    "meeting someone unexpected",
    "the feeling of learning something new",
    "a childhood memory that suddenly came back",
    "exploring a new place for the first time",
    "a conversation that changed your perspective",
    "the best part of your morning routine",
    "watching rain from a window",
    "finding something you thought you had lost",
    "the taste of food that reminds you of home",
    "a walk through a crowded market",
    "waiting for something important",
    "the moment before falling asleep",
    "helping a stranger",
    "discovering a hidden spot in your neighborhood",
    "a song that brings back strong memories",
    "the smell of fresh bread in the morning",
    "watching people in a café",
    "the first day of something new",
    "saying goodbye at a train station",
]

# ── Content generation prompt ──────────────────────────────────────

IMMERSION_SYSTEM_PROMPT = """You are an English immersion content generator. Your ONLY job is to output a valid JSON object. Nothing else.

## ABSOLUTE REQUIREMENT
You MUST output ONLY a single JSON object on one line, with NO markdown fences, NO introductions, NO explanations, NO extra text before or after. Just the JSON.

## Content rules
- Write a short, vivid English passage (80-150 words) for LISTENING practice.
- Write naturally — the way a native speaker would tell a story.
- Be concrete and sensory: describe what you SEE, HEAR, SMELL, FEEL, TASTE.
- Include 3-5 key vocabulary items that repeat naturally 2-3 times.
- Make it emotionally engaging — surprise, wonder, nostalgia, curiosity, joy.
- NO Chinese characters, NO pinyin, NO translations.

## Difficulty: {difficulty}
{level_guide}

## Topic seed: {topic}

## Content type: {content_type}
{type_guide}

## REQUIRED JSON FORMAT (your entire response must be exactly this structure):
{{"title": "Short engaging title (4-8 words)", "key_words": ["word1", "word2", "word3"], "passage": "Full passage text 80-150 words. Use vivid sensory English. Escape double quotes inside with backslash.", "closing_thought": "One reflective question (1 sentence)."}}"""

TYPE_GUIDES: dict[str, str] = {
    "daily_scene": "Describe a vivid moment from everyday life — making coffee, walking in the rain, cooking dinner, watching the sunset. Focus on sensory details.",
    "mini_story": "Tell a very short story with a tiny emotional arc — a small surprise, a moment of joy, an unexpected kindness. Beginning, middle, and gentle end.",
    "place_description": "Describe a real or imaginary place in rich sensory detail. What does it look like, sound like, smell like? Make the listener feel they are THERE.",
    "dialogue": "Write a short natural conversation between two people. Use quotation marks. Show personality through how they speak.",
    "observation": "A first-person reflection on something interesting noticed today. Intimate, thoughtful, like a journal entry.",
}


@dataclass
class ImmersionContent:
    """A single piece of immersion content."""
    title: str
    content_type: str
    difficulty: str
    key_words: list[str]
    passage: str
    closing_thought: str


# ── Immersion Session ──────────────────────────────────────────────

IMMERSION_HELP = """[bold cyan]Immersion Commands:[/bold cyan]
  [bold]Enter[/bold] / [bold]n[/bold]   Next → generate new content
  [bold]a[/bold]           Again — replay from Pass 1
  [bold]w[/bold]           Show key vocabulary
  [bold]d[/bold]           Change difficulty (easy / medium / hard)
  [bold]t[/bold]           Set a specific topic
  [bold]- / +[/bold]       Adjust speech speed
  [bold]s[/bold]           Skip this content
  [bold]?[/bold]           Show this help
  [bold]q[/bold]           Quit immersion mode"""


class ImmersionSession:
    """Manages a multi-pass English immersion experience.

    Three-pass approach per content piece:
      1. Sound First  — Listen only, no text. Pure auditory input.
      2. Read Along   — See text while listening. Connect sound→text.
      3. Eyes Closed  — Audio only again. Consolidate direct pathways.
    """

    def __init__(self, client: LLMClient, difficulty: str = "medium"):
        self.client = client
        self.difficulty = difficulty
        self.speech_rate: int = 0  # offset from default, range -20 to +20
        self.topic: str | None = None
        self.content_history: list[str] = []  # avoid repeating topics
        self.current_content: ImmersionContent | None = None
        self.session_count: int = 0

    async def generate_content(self, content_type: str | None = None) -> ImmersionContent:
        """Generate a new piece of immersion content via LLM."""
        if content_type is None:
            content_type = random.choice(CONTENT_TYPES)

        topic_seed = self.topic or random.choice(TOPIC_SEEDS)

        prompt = IMMERSION_SYSTEM_PROMPT.format(
            difficulty=self.difficulty,
            level_guide=DIFFICULTY_DESCRIPTIONS.get(self.difficulty, ""),
            topic=topic_seed,
            content_type=content_type,
            type_guide=TYPE_GUIDES.get(content_type, ""),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Generate a {CONTENT_TYPE_LABELS.get(content_type, content_type)} "
                f"immersion passage at {self.difficulty} level. "
                f"Topic: {topic_seed}. Output ONLY the JSON object, nothing else."
            )},
        ]

        try:
            response = await self.client.chat_sync(
                messages, temperature=0.7, max_tokens=512,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            console.print(f"[red]Failed to generate content: {e}[/red]")
            raise

        # Parse JSON from LLM response
        data = self._parse_json_response(response)

        # Clean passage text: strip any markdown or artifacts
        passage = self._clean_passage(data.get("passage", response))

        content = ImmersionContent(
            title=data.get("title", "Listening Passage"),
            content_type=content_type,
            difficulty=self.difficulty,
            key_words=data.get("key_words", []),
            passage=passage,
            closing_thought=data.get("closing_thought", "What did you notice in what you heard?"),
        )

        self.current_content = content
        self.content_history.append(content.title)
        self.session_count += 1
        return content

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Extract and parse JSON from an LLM response that may have markdown fences."""
        text = response.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json fence
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass

        # Try extracting from any ``` fence
        if "```" in text:
            try:
                start = text.index("```") + 3
                # Skip language tag if present
                nl = text.index("\n", start) if "\n" in text[start:start+20] else start
                if nl - start < 15:
                    start = nl + 1
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass

        # Try finding { } block
        try:
            brace_start = text.index("{")
            brace_end = text.rindex("}") + 1
            return json.loads(text[brace_start:brace_end])
        except (ValueError, json.JSONDecodeError):
            pass

        console.print("[yellow]Warning: Could not parse JSON from LLM response, using raw text.[/yellow]")
        return {"passage": text}

    @staticmethod
    def _clean_passage(text: str) -> str:
        """Clean up passage text by removing markdown artifacts and excessive whitespace."""
        # Remove markdown code fences if present
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            # Skip lines that are just markdown fences
            stripped = line.strip()
            if stripped in ("```", "```json", "```markdown"):
                continue
            # Skip lines that look like JSON fragments (leftover from bad parsing)
            if stripped.startswith('{"') and stripped.endswith('}'):
                continue
            cleaned.append(line)
        result = "\n".join(cleaned).strip()
        # Remove leading/trailing quotes that might wrap the entire passage
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        return result

    async def run(self):
        """Run the immersion session loop."""
        self._print_welcome()

        # Breathing guide — switches brain to parasympathetic (learning) mode
        await self._breathing_guide()

        # Generate first piece of content
        try:
            console.print("[dim]✨ Generating your first immersion passage...[/dim]")
            await self.generate_content()
        except Exception:
            console.print("[red]Failed to start immersion mode. Check your API connection.[/red]")
            return

        while True:
            await self._play_content()

            # Prompt for next action
            console.print()
            choice = Prompt.ask(
                "[dim]Enter=next  a=again  w=words  d=difficulty  t=topic  s=skip  ?=help  q=quit[/dim]",
                default="n",
            ).strip().lower()

            if choice in ("q", "quit", "/back", "/quit"):
                self._print_summary()
                await self._quiet_period_reminder()
                break
            elif choice in ("?", "help", "/help"):
                console.print()
                console.print(IMMERSION_HELP)
            elif choice in ("a", "again", "r", "repeat"):
                # Replay current content from Pass 1
                pass
            elif choice in ("w", "words", "/words"):
                self._show_key_words()
                continue
            elif choice in ("s", "skip"):
                try:
                    console.print("[dim]Generating new content...[/dim]")
                    content_type = random.choice(CONTENT_TYPES)
                    await self.generate_content(content_type)
                except Exception:
                    console.print("[red]Failed to generate. Try again.[/red]")
            elif choice in ("t", "topic"):
                new_topic = Prompt.ask("What topic? (e.g. 'walking in a forest', 'cooking dinner')", default="")
                if new_topic.strip():
                    self.topic = new_topic.strip()
                    console.print(f"[green]Topic: {self.topic}[/green]")
                else:
                    self.topic = None
                    console.print("[dim]Topic cleared — using random seeds.[/dim]")
                try:
                    await self.generate_content()
                except Exception:
                    console.print("[red]Failed to generate. Try again.[/red]")
            elif choice in ("d", "difficulty"):
                console.print()
                for key, label in DIFFICULTY_LABELS.items():
                    marker = " [bold green]← current[/bold green]" if key == self.difficulty else ""
                    console.print(f"  [bold]{key}[/bold] — {label}{marker}")
                new_diff = Prompt.ask("Choose difficulty", default=self.difficulty).strip().lower()
                if new_diff in DIFFICULTY_LABELS:
                    self.difficulty = new_diff
                    console.print(f"[green]Difficulty: {DIFFICULTY_LABELS[new_diff]}[/green]")
                    try:
                        await self.generate_content()
                    except Exception:
                        console.print("[red]Failed to generate. Try again.[/red]")
                else:
                    console.print(f"[yellow]Keep current: {self.difficulty}[/yellow]")
            elif choice in ("-", "slower"):
                self.speech_rate = max(-30, self.speech_rate - 5)
                console.print(f"[dim]Speech rate offset: {self.speech_rate} (slower)[/dim]")
                # Replay current pass audio at new rate
                if self.current_content:
                    speak_now(self.current_content.passage)
            elif choice in ("+", "faster"):
                self.speech_rate = min(30, self.speech_rate + 5)
                console.print(f"[dim]Speech rate offset: {self.speech_rate} (faster)[/dim]")
                if self.current_content:
                    speak_now(self.current_content.passage)
            else:
                # Default (Enter, n, next, empty): generate new content
                try:
                    content_type = random.choice(CONTENT_TYPES)
                    await self.generate_content(content_type)
                except Exception:
                    console.print("[red]Failed to generate. Try again.[/red]")

    async def _play_content(self):
        """Execute the 3-pass playback for current content."""
        c = self.current_content
        if not c:
            return

        # ── Pass 1: Sound First (audio only, no text displayed) ──
        console.print()
        console.print(Rule(style="dim"))
        console.print(Panel(
            "[bold]Pass 1/3 — 👂 Sound First[/bold]\n"
            "[dim]Listen only. Don't try to understand every word.\n"
            "Let the rhythm and intonation wash over you.\n"
            "Notice: Is the speaker happy? Thoughtful? Surprised?\n\n"
            "💡 Relax your shoulders. Unclench your jaw.\n"
            "   You're not being tested. You're just listening.[/dim]",
            border_style="yellow",
            width=WIDTH,
        ))
        console.print(f"[yellow]🎧 Now playing...[/yellow]")
        speak_now(c.passage)
        console.print()
        console.print("[green]✓ Done[/green] [dim]— What did you feel? Any words stand out?[/dim]")
        self._wait_for_enter("Press Enter for Pass 2")

        # ── Pass 2: Read Along (text + audio together) ──
        console.print()
        console.print(Panel(
            "[bold]Pass 2/3 — 👁️ Read Along[/bold]\n"
            "[dim]Now see the text AND listen together.\n"
            "Connect the written words to the sounds you heard.\n"
            "Notice: Did anything sound different than you expected?[/dim]",
            border_style="cyan",
            width=WIDTH,
        ))

        console.print()
        console.print(Panel(
            Markdown(c.passage),
            title=f"[bold cyan]{c.title}[/bold cyan]  ·  {CONTENT_TYPE_LABELS.get(c.content_type, c.content_type)}  ·  {DIFFICULTY_LABELS.get(c.difficulty, c.difficulty)}",
            border_style="cyan",
            width=WIDTH,
        ))

        console.print("[yellow]🔊 Playing while you read...[/yellow]")
        speak_now(c.passage)
        console.print("[green]✓ Done[/green] [dim]— Did the words look like you imagined?[/dim]")
        self._wait_for_enter("Press Enter for Pass 3")

        # ── Pass 3: Eyes Closed (audio only, consolidate) ──
        console.print()
        console.print(Panel(
            "[bold]Pass 3/3 — 🧘 Eyes Closed[/bold]\n"
            "[dim]Close your eyes. Just listen one more time.\n"
            "Your brain is building direct sound→meaning pathways.\n"
            "No translation needed — the meaning should start to feel natural.[/dim]",
            border_style="magenta",
            width=WIDTH,
        ))
        console.print("[yellow]🎧 Final listen...[/yellow]")
        speak_now(c.passage)
        console.print()
        console.print("[green]✓ Done[/green] [dim]— How much more did you understand this time?[/dim]")

        # ── After the 3 passes: reveal key words + closing thought ──
        self._show_key_words()
        console.print(f"[italic dim]💭 {c.closing_thought}[/italic dim]")

        # ── Mini celebration every 5 sessions ──
        if self.session_count > 0 and self.session_count % 5 == 0:
            console.print()
            console.print(Panel(
                f"[bold green]🎉 You've completed {self.session_count} immersion sessions![/bold green]\n"
                "[dim]That's like {0} minutes of pure English input. Your brain is adapting.[/dim]".format(
                    self.session_count * 3
                ),
                border_style="green",
                width=WIDTH,
            ))

    def _show_key_words(self):
        """Display key vocabulary for current content."""
        c = self.current_content
        if not c:
            return
        if not c.key_words:
            return

        # Build a compact table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold green")
        for w in c.key_words:
            table.add_row(f"• {w}")

        console.print()
        console.print(Panel(
            table,
            title="🔑 Key Vocabulary (notice how they appeared in context)",
            border_style="green",
            width=WIDTH,
        ))

    def _wait_for_enter(self, prompt_text: str = "Press Enter to continue"):
        """Wait for the user to press Enter."""
        try:
            Prompt.ask(f"[dim]{prompt_text}[/dim]", default="")
        except (EOFError, KeyboardInterrupt):
            pass

    def _print_welcome(self):
        """Show the immersion mode welcome screen."""
        console.clear()
        console.print()
        console.print(Panel(
            "[bold cyan]🌊  English Immersion Mode[/bold cyan]\n"
            "[dim]Phase-1 Pretraining Input — the learning phase most adult learners skip[/dim]\n\n"
            "This mode helps you build [bold]direct sound→meaning connections[/bold]\n"
            "by removing the translation step. Just listen and absorb.\n\n"
            "[dim italic]\"You can't learn to swim by reading about water.\"\n"
            "You have to get in. This is your pool.[/dim italic]",
            border_style="cyan",
            width=WIDTH,
        ))
        console.print()
        console.print(IMMERSION_HELP)
        console.print()
        console.print(f"[dim]Difficulty: {DIFFICULTY_LABELS[self.difficulty]}  |  "
                      f"Speech speed: default[/dim]")
        console.print()

    async def _breathing_guide(self):
        """Guide the user through 30 seconds of deep breathing.

        Extended exhale (4s in → 2s hold → 6s out) activates the vagus nerve
        and shifts the nervous system into parasympathetic (rest/digest) mode.
        Procedural memory (basal ganglia) only works in this state — the
        sympathetic 'fight/flight' state locks you into declarative memory
        (hippocampus), which is exactly what we're trying to bypass.
        """
        console.print()
        console.print(Panel(
            "[bold]🧘  Before We Begin — 3 Deep Breaths[/bold]\n\n"
            "[dim]Take a moment to switch your brain into learning mode.\n"
            "We'll do 3 deep breaths together — about 30 seconds.\n\n"
            "This activates your [bold]parasympathetic nervous system[/bold]:\n"
            "the state where real language absorption happens.\n"
            "Not memorization. Not analysis. Pure absorption.[/dim]",
            border_style="blue",
            width=WIDTH,
        ))

        for i in range(1, 4):
            console.print(f"\n[bold blue]Breath {i}/3[/bold blue]")
            console.print("[dim]  Breathe in  through your nose... (4 seconds)[/dim]")
            await asyncio.sleep(1)
            for sec in range(4, 0, -1):
                console.print(f"  [cyan]↗ In  {sec}...[/cyan]")
                await asyncio.sleep(1)
            console.print("[dim]  Hold... (2 seconds)[/dim]")
            await asyncio.sleep(2)
            console.print("[dim]  Breathe out through your mouth... (6 seconds)[/dim]")
            for sec in range(6, 0, -1):
                console.print(f"  [cyan]↘ Out {sec}...[/cyan]")
                await asyncio.sleep(1)

        console.print()
        console.print("[green]✓ Ready.[/green] Your brain is now in [bold]learning mode[/bold]. Let's begin.\n")
        await asyncio.sleep(0.5)

    async def _quiet_period_reminder(self):
        """After immersion, remind the user to protect the learning state.

        The brain needs time to transfer what was absorbed from working
        buffers into longer-term consolidation pathways. Immediately
        flooding it with Chinese or social media overwrites the weak
        English traces before they can be stabilized.
        """
        console.print()
        console.print(Panel(
            "[bold]🤫  2-Minute Quiet Period[/bold]\n\n"
            "[dim]Your brain just processed a lot of English input.\n"
            "Don't rush to fill it with something else.\n\n"
            "If possible:\n"
            "  • Don't open your phone or social media\n"
            "  • Don't switch to Chinese content\n"
            "  • Just sit quietly, or take a short walk\n"
            "  • Let your mind wander — daydreaming helps consolidation\n\n"
            "This gives your brain time to move what you absorbed\n"
            "from short-term buffers into longer-term storage.[/dim]",
            border_style="blue",
            width=WIDTH,
        ))

    def _print_summary(self):
        """Print a summary of the immersion session."""
        console.print()
        console.print(Panel(
            f"[bold green]🌊 Immersion Complete![/bold green]\n\n"
            f"Sessions: [bold]{self.session_count}[/bold]\n"
            f"Estimated listening time: ~[bold]{self.session_count * 3}[/bold] minutes\n"
            f"Difficulty: [bold]{DIFFICULTY_LABELS[self.difficulty]}[/bold]\n\n"
            "[dim]Remember: consistency > intensity.\n"
            "A little every day builds the pathway. 🧠✨[/dim]\n\n"
            "[bold]💡 Best times for immersion:[/bold]\n"
            "[dim]  🌙 Right before sleep → immediate REM consolidation\n"
            "  🌅 Right after waking → pre-frontal cortex still offline,\n"
            "      brain state closest to an infant's[/dim]",
            border_style="green",
            width=WIDTH,
        ))


# ── Public entry point ─────────────────────────────────────────────

async def immersion_session(
    client: LLMClient,
    topic: str | None = None,
    difficulty: str = "medium",
):
    """Launch an English immersion session.

    Args:
        client: LLM client for content generation.
        topic: Optional topic seed (e.g. "cooking", "travel").
        difficulty: One of "easy", "medium", "hard".
    """
    if difficulty not in DIFFICULTY_LABELS:
        console.print(f"[yellow]Unknown difficulty '{difficulty}', using 'medium'.[/yellow]")
        difficulty = "medium"

    session = ImmersionSession(client, difficulty=difficulty)
    if topic:
        session.topic = topic
    await session.run()
