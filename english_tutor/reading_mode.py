"""Reading mode — browse, read, and learn from daily English articles."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from english_tutor.content_loader import (
    Article,
    FeedFetcher,
    extract_new_words,
    grade_difficulty,
)
from english_tutor.llm_client import LLMClient
from english_tutor.memory import ConversationMemory
from english_tutor.spaced_repetition import SpacedRepetition
from english_tutor.tts import speak_now
from english_tutor.tutor_prompt import build_system_message

console = Console()
WIDTH = 78

READING_COMMANDS = {
    "/help": "Show reading mode commands",
    "/list": "List all available articles",
    "/summary": "AI summary of the current article",
    "/words": "Show new words found in this article",
    "/define": "Define a word: /define <word>",
    "/speak": "Read the current article aloud",
    "/back": "Return to conversation mode",
    "/quit": "Exit the tutor",
}


def print_commands():
    console.print("\n[bold]Reading commands:[/bold]")
    for cmd, desc in READING_COMMANDS.items():
        console.print(f"  [bold]{cmd:<12}[/bold] {desc}")


def print_article_list(articles: list[Article]):
    """Display a table of available articles."""
    if not articles:
        console.print("[dim]No articles available. Try fetching first with /read.[/dim]")
        return

    table = Table(
        title="📰 Today's Articles",
        box=None,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", width=45)
    table.add_column("Level", width=12)
    table.add_column("Words", width=6)
    table.add_column("Source", width=16)

    for i, a in enumerate(articles, 1):
        level_style = {
            "beginner": "green",
            "intermediate": "yellow",
            "advanced": "red",
            "unknown": "dim",
        }.get(a.difficulty, "dim")

        table.add_row(
            str(i),
            a.title[:45],
            f"[{level_style}]{a.difficulty}[/{level_style}]",
            str(a.word_count),
            a.source[:16],
        )

    console.print(table)
    console.print("\n[dim]Type a number to read an article, or type a command.[/dim]")


def print_article(article: Article, page: int = 0):
    """Display an article in paginated format."""
    if not article.text:
        console.print("[red]No article content loaded.[/red]")
        return

    # Split into paragraphs
    paragraphs = [p.strip() for p in article.text.split("\n\n") if p.strip()]

    # Pagination
    PARAGRAPHS_PER_PAGE = 5
    total_pages = max(1, (len(paragraphs) + PARAGRAPHS_PER_PAGE - 1) // PARAGRAPHS_PER_PAGE)
    page = min(page, total_pages - 1)
    start = page * PARAGRAPHS_PER_PAGE
    end = start + PARAGRAPHS_PER_PAGE
    current_pars = paragraphs[start:end]

    # Header
    console.print()
    console.print(f"[bold cyan]{article.title}[/bold cyan]")
    console.print(f"[dim]{article.source} • {article.word_count} words • {article.difficulty}[/dim]")
    if article.new_words:
        new_word_list = ", ".join(w["word"] for w in article.new_words[:8])
        console.print(f"[green]New words: {new_word_list}[/green]")
    console.print(Rule(style="dim"))

    # Paragraphs
    for i, par in enumerate(current_pars, start + 1):
        console.print(f"\n[dim]{i:>3}[/dim] {par}")
        console.print()

    # Page indicator
    if total_pages > 1:
        console.print(f"[dim]Page {page + 1}/{total_pages} — [/dim]", end="")
        console.print("[dim]scroll with /more, or go back with /list[/dim]")

    console.print(Rule(style="dim"))


# ── Reading session ────────────────────────────────────────────────

async def reading_session(
    llm_client: LLMClient,
    memory: ConversationMemory,
    sr: SpacedRepetition,
):
    """Run the reading mode — fetch, browse, and read articles interactively."""
    fetcher = FeedFetcher()

    try:
        # Try cache first, then fetch
        articles = fetcher.get_cached_articles(max_age_hours=6)
        if not articles:
            console.print("[cyan]Fetching today's articles...[/cyan]")
            articles = await fetcher.fetch_feeds()
            if not articles:
                console.print("[red]No articles fetched. Check your internet connection.[/red]")
                return

            # Extract content for each article
            console.print("[cyan]Extracting article content...[/cyan]")
            for i, article in enumerate(articles):
                console.print(f"  [{i+1}/{len(articles)}] {article.title[:50]}...")
                article = await fetcher.extract_content(article)
                article.difficulty = grade_difficulty(article.text)
                article.new_words = extract_new_words(
                    article.text,
                    {w["word"].lower() for w in memory.get_vocabulary()},
                )
                articles[i] = article

            fetcher.cache_articles(articles)
            console.print(f"[green]✓ {len(articles)} articles ready![/green]")
        else:
            console.print(f"[dim]Loaded {len(articles)} articles from cache.[/dim]")

        # Browse loop
        current_article: Article | None = None
        current_page = 0

        print_article_list(articles)

        while True:
            try:
                cmd = Prompt.ask("\n[bold yellow]Reading[/bold yellow]")
            except (EOFError, KeyboardInterrupt):
                break

            cmd = cmd.strip().lower()

            # ── Navigate to article by number ──────────────────
            if cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(articles):
                    current_article = articles[idx]
                    current_page = 0
                    print_article(current_article)
                else:
                    console.print(f"[red]Invalid number. Choose 1-{len(articles)}.[/red]")
                continue

            # ── List articles ──────────────────────────────────
            if cmd == "/list":
                print_article_list(articles)
                continue

            # ── Next page ──────────────────────────────────────
            if cmd in ("/more", "/next"):
                if current_article:
                    paragraphs = [p.strip() for p in current_article.text.split("\n\n") if p.strip()]
                    total = max(1, (len(paragraphs) + 4) // 5)
                    current_page = min(current_page + 1, total - 1)
                    print_article(current_article, current_page)
                else:
                    console.print("[dim]Select an article first with its number.[/dim]")
                continue

            # ── Previous page ──────────────────────────────────
            if cmd == "/prev":
                if current_article:
                    current_page = max(0, current_page - 1)
                    print_article(current_article, current_page)
                else:
                    console.print("[dim]Select an article first with its number.[/dim]")
                continue

            # ── New words ──────────────────────────────────────
            if cmd == "/words":
                if current_article and current_article.new_words:
                    table = Table(title="📖 New Words", box=None, header_style="bold cyan")
                    table.add_column("Word", style="bold")
                    table.add_column("Context", max_width=50)
                    for w in current_article.new_words[:15]:
                        table.add_row(w["word"], w["context"][:50])
                    console.print(table)
                    console.print("\n[dim]Use /define <word> to get a definition.[/dim]")
                else:
                    console.print("[dim]No new words identified yet. Read an article first.[/dim]")
                continue

            # ── Define a word ──────────────────────────────────
            if cmd.startswith("/define"):
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /define <word>[/yellow]")
                    continue
                word = parts[1].strip()
                console.print(f"[cyan]Looking up '{word}'...[/cyan]")
                try:
                    definition = await llm_client.chat_sync(
                        [
                            {
                                "role": "system",
                                "content": "You are a dictionary. Give a brief definition (1 sentence), "
                                "pronunciation, and one example sentence for the word provided. "
                                "Keep it concise.",
                            },
                            {"role": "user", "content": f"Define '{word}' for an English learner at CET-4 level."},
                        ],
                        max_tokens=150,
                        temperature=0.3,
                    )
                    console.print(f"\n[bold]{word}[/bold]")
                    console.print(definition)
                    # Optionally save to vocabulary
                    save = Prompt.ask(
                        f"Save '[bold]{word}[/bold]' to your vocabulary?", choices=["y", "n"], default="y"
                    )
                    if save == "y":
                        memory.add_new_word(word, definition[:100], "")
                        sr.add_card(word, definition[:100], current_article.title if current_article else "")
                        console.print(f"[green]✓ Saved '{word}' to vocabulary + spaced repetition![/green]")
                except Exception as e:
                    console.print(f"[red]Error looking up word: {e}[/red]")
                continue

            # ── AI Summary ─────────────────────────────────────
            if cmd == "/summary":
                if current_article:
                    console.print("[cyan]Generating summary...[/cyan]")
                    try:
                        summary = await llm_client.chat_sync(
                            [
                                build_system_message(),
                                {
                                    "role": "user",
                                    "content": (
                                        f"Please summarize this article for a CET-4 level English learner. "
                                        f"Use simple English. Include: main topic, key points (3-4 bullet points), "
                                        f"and 3 new vocabulary words with definitions.\n\n"
                                        f"Title: {current_article.title}\n\n{current_article.text[:2000]}"
                                    ),
                                },
                            ],
                            max_tokens=500,
                            temperature=0.5,
                        )
                        console.print(Panel(
                            Markdown(summary),
                            title=f"📝 Summary: {current_article.title[:40]}",
                            border_style="cyan",
                            width=WIDTH,
                        ))
                    except Exception as e:
                        console.print(f"[red]Error generating summary: {e}[/red]")
                else:
                    console.print("[dim]Select an article first with its number.[/dim]")
                continue

            # ── Speak article aloud ──────────────────────────────
            if cmd == "/speak":
                if current_article and current_article.text:
                    # Speak first 2000 chars (SPD limit)
                    text = current_article.text[:2000]
                    console.print(f"[cyan]🔊 Reading '{current_article.title[:40]}' aloud...[/cyan]")
                    speak_now(text)
                    console.print("[green]✓ Done[/green]")
                else:
                    console.print("[dim]Select an article first with its number.[/dim]")
                continue

            # ── Back to conversation ───────────────────────────
            if cmd == "/back":
                console.print("[green]Returning to conversation mode...[/green]")
                return

            # ── Help ───────────────────────────────────────────
            if cmd == "/help":
                print_commands()
                continue

            # ── Quit ───────────────────────────────────────────
            if cmd in ("/quit", "/exit"):
                return

            console.print("[dim]Unknown command. Type /help to see available commands.[/dim]")

    finally:
        await fetcher.close()
