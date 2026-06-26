#!/usr/bin/env python3
"""English Tutor CLI — Rich-powered interactive English learning assistant."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from english_tutor.llm_client import LLMClient
from english_tutor.memory import ConversationMemory
from english_tutor.reading_mode import reading_session
from english_tutor.review_mode import review_session, quiz_session
from english_tutor.spaced_repetition import SpacedRepetition
from english_tutor.stt import record_and_transcribe, is_recording_available
from english_tutor.tts import speak_now
from english_tutor.tutor_prompt import (
    build_system_message,
    build_topic_suggestion,
    CONVERSATION_SUMMARY_PROMPT,
)

load_dotenv()

console = Console()
WIDTH = 78

# ── Helpers ─────────────────────────────────────────────────────────

BANNER = r"""
[bold cyan]
   _____ _ _   _        _____    _              _
  | ____(_) | | |_   _ |_   _|__| |_ _ __ ___ (_)_ __
  |  _| |_| |_| | | | | | |/ _ \ __| '_ ` _ \| | '_ \
  | |___| |  _  | |_| | | |  __/ |_| | | | | | | | | |
  |_____|_|_| |_|\__,_| |_|\___|\__|_| |_| |_|_|_| |_|
[/bold cyan]
[dim]Your AI English Tutor — practice daily, improve naturally[/dim]
"""

COMMANDS = {
    "/help": "Show this help message",
    "/new": "Start a new conversation session",
    "/stats": "Show your learning statistics",
    "/words": "Show your vocabulary list",
    "/errors": "Show common errors with practice suggestions",
    "/review": "Review due vocabulary cards (spaced repetition)",
    "/quiz": "Quick vocabulary quiz",
    "/topic": "Suggest a new conversation topic",
    "/read": "Fetch and read today's English articles",
    "/speak": "Read the last response aloud (TTS)",
    "/record": "Record voice and transcribe (e.g. /record 5)",
    "/save": "Save and exit",
    "/quit": "Exit the tutor",
}

HELP_TEXT = "\n".join(f"  [bold]{cmd}[/bold]    {desc}" for cmd, desc in COMMANDS.items())


def print_welcome():
    console.clear()
    console.print(BANNER)
    console.print(Rule(style="dim"))
    console.print(f"\n[dim]Type anything to start chatting with Emma, your English tutor.\nType [bold]/help[/bold] to see available commands.[/dim]\n")


def print_message(role: str, content: str):
    """Display a message with appropriate styling."""
    if role == "assistant":
        label = Text.assemble(("Emma", "bold green"), (" 💬", "dim"))
        panel_style = "green"
    elif role == "system":
        label = Text("ℹ️", "dim")
        panel_style = "dim"
    else:
        label = Text("You", "bold yellow")
        panel_style = "yellow"

    console.print()
    console.print(Panel(
        Markdown(content),
        title=label,
        border_style=panel_style,
        width=WIDTH,
        title_align="left",
    ))
    console.print()


def print_words_table(words: list[dict]):
    """Display vocabulary in a table."""
    if not words:
        console.print("[dim]No words saved yet. Start a conversation to learn new words![/dim]")
        return

    table = Table(title="📖 Your Vocabulary", box=None, header_style="bold cyan")
    table.add_column("Word", style="bold")
    table.add_column("Definition")
    table.add_column("Context", max_width=30)
    for w in words[-20:]:  # Show last 20
        table.add_row(
            w.get("word", ""),
            w.get("definition", "")[:40],
            w.get("context", "")[:30],
        )
    console.print(table)


def print_stats(memory: ConversationMemory, sr: SpacedRepetition | None = None):
    """Display learning stats."""
    sr_stats = sr.get_stats() if sr else None
    console.print(Panel(
        Markdown(memory.get_stats_summary(sr_stats)),
        title="📊 Progress",
        border_style="cyan",
        width=WIDTH,
    ))


def print_errors(memory: ConversationMemory):
    """Display detailed error analysis with practice suggestions."""
    errors = memory.stats.get("common_errors", {})
    if not errors:
        console.print(Panel(
            "[green]No common errors recorded yet! Keep practicing and Emma will help track them.[/green]",
            title="✅ Error Analysis",
            border_style="green",
            width=WIDTH,
        ))
        return

    # Error type descriptions and practice suggestions
    error_guides = {
        "tense": {
            "description": "Verb tense confusion (past/present/future)",
            "practice": "Try describing what you did yesterday (past tense), then what you do every day (present tense).",
        },
        "article": {
            "description": "Article usage (a/an/the)",
            "practice": "Practice: 'I saw _ dog. _ dog was brown.' → 'I saw a dog. The dog was brown.'",
        },
        "preposition": {
            "description": "Preposition errors (in/on/at/to/for)",
            "practice": "Practice: 'I go _ school _ bus.' → 'I go to school by bus.'",
        },
        "word_order": {
            "description": "Word order / sentence structure",
            "practice": "English follows Subject-Verb-Object order. Practice: 'Yesterday go I to store' → 'Yesterday I went to the store.'",
        },
        "vocabulary": {
            "description": "Word choice / collocation errors",
            "practice": "Try reading more to see words in context. Pay attention to which words naturally go together.",
        },
        "plural": {
            "description": "Singular/plural agreement",
            "practice": "Practice: 'I have three book' → 'I have three books.' Most nouns add -s or -es.",
        },
        "subject_verb": {
            "description": "Subject-verb agreement",
            "practice": "Practice: 'He go to school' → 'He goes to school.' Third person singular adds -s.",
        },
    }

    lines = ["## 📝 Common Error Patterns\n"]
    total = sum(v["count"] for v in errors.values())
    lines.append(f"**Total errors tracked: {total}**\n")

    # Sort by frequency
    sorted_errors = sorted(errors.items(), key=lambda x: x[1]["count"], reverse=True)

    for error_type, data in sorted_errors:
        guide = error_guides.get(error_type, {
            "description": error_type.replace("_", " ").title(),
            "practice": "Review this grammar point and practice with Emma.",
        })
        count = data["count"]
        bar = "█" * min(count, 20) + "░" * max(0, 20 - min(count, 20))
        lines.append(f"  **{error_type.title()}** ({count}x)")
        lines.append(f"  `{bar}` _{guide['description']}_")
        lines.append(f"  💡 {guide['practice']}")
        if data["examples"]:
            examples = data["examples"][-3:]
            lines.append(f"  📌 Example{'s' if len(examples) > 1 else ''}:")
            for ex in examples:
                lines.append(f"    > _{ex}_")
        lines.append("")

    lines.append("---")
    lines.append("💡 **Tip:** Practice these patterns in conversation with Emma. ")
    lines.append("She will gently correct you and help you improve!")

    console.print(Panel(
        Markdown("\n".join(lines)),
        title="📝 Error Analysis",
        border_style="yellow",
        width=WIDTH,
    ))


def print_thinking():
    """Show a 'thinking' indicator."""
    console.print("[dim]Emma is thinking...[/dim]")


# ── Main conversation loop ──────────────────────────────────────────

async def chat_loop(client: LLMClient, memory: ConversationMemory, sr: SpacedRepetition):
    """Run the interactive chat session."""

    # Build initial messages with system prompt
    system_msg = build_system_message()
    messages = [system_msg]

    # Add conversation history if resuming
    for msg in memory.get_context(max_messages=20):
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # If this is a new session, have Emma start the conversation
    last_response = ""
    if len(messages) == 1:
        greeting = (
            "Hi there! I'm Emma, your English tutor. 😊 "
            "How are you today? What would you like to talk about?\n\n"
            "Here are some ideas if you're not sure:\n"
            f"- {build_topic_suggestion()}"
        )
        messages.append({"role": "assistant", "content": greeting})
        memory.save_message("assistant", greeting)
        last_response = greeting
        print_message("assistant", greeting)

    while True:
        # ── Get user input ──────────────────────────────────────
        try:
            user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]See you next time! Keep practicing![/yellow]")
            break

        user_input = user_input.strip()

        # ── Handle commands ──────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit":
                console.print("[green]Goodbye! Keep up the great work! 🎉[/green]")
                break
            elif cmd == "/help":
                console.print(Panel(HELP_TEXT, title="Commands", border_style="dim", width=WIDTH))
                continue
            elif cmd == "/new":
                memory.new_session()
                messages = [system_msg]
                console.print("[green]Started a fresh conversation![/green]")
                continue
            elif cmd == "/stats":
                print_stats(memory, sr)
                continue
            elif cmd == "/words":
                print_words_table(memory.get_vocabulary())
                continue
            elif cmd == "/topic":
                topic = build_topic_suggestion(set(memory.topic_history))
                memory.topic_history.append(topic)
                console.print(f"[green]How about this topic:[/green] {topic}")
                continue
            elif cmd == "/save":
                console.print("[green]Session saved![/green]")
                continue
            elif cmd == "/errors":
                print_errors(memory)
                continue
            elif cmd == "/review":
                await review_session(sr)
                continue
            elif cmd == "/quiz":
                await quiz_session(sr)
                continue
            elif cmd == "/speak":
                if last_response:
                    console.print("[cyan]🔊 Speaking...[/cyan]")
                    speak_now(last_response)
                    console.print("[green]✓ Done[/green]")
                else:
                    console.print("[yellow]No response to read yet. Start a conversation first![/yellow]")
                continue
            elif cmd.startswith("/record"):
                # Parse optional duration: /record 5
                parts = cmd.split()
                duration = 5
                if len(parts) > 1 and parts[1].isdigit():
                    duration = int(parts[1])

                if not is_recording_available():
                    console.print("[red]Microphone not available (arecord not found).[/red]")
                    continue

                console.print(f"[cyan]🎤 Recording for {duration} seconds... Speak now![/cyan]")
                text = record_and_transcribe(
                    duration=duration, language="en", model_size="tiny"
                )
                if text:
                    console.print(f"[green]You said:[/green] {text}")
                    # Treat as user input — set user_input to continue the loop
                    user_input = text
                else:
                    console.print("[yellow]No speech detected. Try again or type your message.[/yellow]")
                    continue

            elif cmd == "/read":
                console.print("[cyan]Opening reading mode...[/cyan]")
                await reading_session(client, memory, sr)
                console.print("[green]Back to conversation![/green]")
                continue
            else:
                console.print(f"[red]Unknown command: {cmd}. Type /help for available commands.[/red]")
                continue

        if not user_input:
            continue

        # ── Add user message ─────────────────────────────────────
        messages.append({"role": "user", "content": user_input})
        memory.save_message("user", user_input)

        # ── Get streaming response ───────────────────────────────
        print_thinking()
        collected = []
        try:
            async for token in client.chat(messages):
                collected.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
            response = "".join(collected)
            console.print()  # newline after streaming
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            # Remove the user message we added so they can retry
            messages.pop()
            continue

        # ── Save assistant response ──────────────────────────────
        messages.append({"role": "assistant", "content": response})
        memory.save_message("assistant", response)
        last_response = response

        # ── Context window management ────────────────────────────
        # If context gets long, trim older messages but keep system prompt
        MAX_CONTEXT = 40
        if len(messages) > MAX_CONTEXT:
            # Keep system (index 0) + last MAX_CONTEXT-2 exchanges
            keep = [messages[0]] + messages[-(MAX_CONTEXT - 1):]
            messages = keep


# ── Entry point ─────────────────────────────────────────────────────

async def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *_: None)

    # Check for API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        # Try loading from .env in project root
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        console.print("[red]Error: DEEPSEEK_API_KEY not found.[/red]")
        console.print("Create a [bold].env[/bold] file with:")
        console.print("  DEEPSEEK_API_KEY=sk-your-key-here")
        sys.exit(1)

    # Initialize
    client = LLMClient(api_key=api_key)
    memory = ConversationMemory()
    sr = SpacedRepetition()

    # Try to resume the most recent session
    sessions = memory.list_sessions()
    resume_id = None
    for s in sessions:
        if s["message_count"] > 1:  # Has real conversation
            resume_id = s["id"]
            break

    if resume_id:
        memory.load_session(resume_id)
        console.print(f"[dim]📋 Resuming session from {s['date'][:10]} ({s['message_count']} messages)[/dim]")
    else:
        memory.new_session()

    print_welcome()

    # If resuming, show a recap of the last exchange
    if resume_id and len(memory.messages) >= 2:
        last_user = ""
        last_emma = ""
        for msg in reversed(memory.messages):
            if msg["role"] == "assistant" and not last_emma:
                last_emma = msg["content"][:200]
            elif msg["role"] == "user" and not last_user:
                last_user = msg["content"][:200]
            if last_user and last_emma:
                break
        if last_user and last_emma:
            console.print(Panel(
                f"[bold]Your last message:[/bold]\n{last_user}\n\n"
                f"[bold]Emma's last reply:[/bold]\n{last_emma}…",
                title="📝 Where we left off",
                border_style="blue",
                width=80,
            ))

    try:
        await chat_loop(client, memory, sr)
    finally:
        await client.close()
        console.print("[dim]Session saved. See you next time![/dim]")


def run():
    """Entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
