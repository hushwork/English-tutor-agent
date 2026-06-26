"""Review and quiz mode — spaced repetition vocabulary review."""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from english_tutor.spaced_repetition import SpacedRepetition

console = Console()
WIDTH = 78


async def review_session(sr: SpacedRepetition):
    """Interactive spaced repetition review session."""
    due = sr.get_due_cards(limit=20)
    if not due:
        console.print(Panel(
            "🎉 **No cards due for review!**\n\nYou're all caught up. "
            "Keep learning new words in conversation or reading mode.",
            border_style="green",
            width=WIDTH,
        ))
        return

    console.print(f"\n[bold cyan]📚 Review Session — {len(due)} cards due[/bold cyan]")
    console.print("[dim]For each word, try to recall its meaning, then rate your recall.[/dim]\n")

    reviewed = 0
    for card in due:
        reviewed += 1

        # Show the word
        console.print(f"\n[bold]Word {reviewed}/{len(due)}:[/bold] [bold yellow]{card.word}[/bold yellow]")
        if card.context:
            console.print(f"[dim]Context: {card.context[:80]}[/dim]")

        console.print("[dim]Press Enter to reveal the definition...[/dim]")
        try:
            Prompt.ask("")
        except (EOFError, KeyboardInterrupt):
            break

        # Reveal definition
        console.print(f"\n[green]Definition:[/green] {card.definition}")

        # Ask for quality rating
        console.print("\n[dim]How well did you remember?[/dim]")
        console.print("  [red]1-2[/red] = Forgotten/Hard   [yellow]3[/yellow] = Hesitant   [green]4-5[/green] = Good/Perfect")

        try:
            rating = Prompt.ask("Your rating", choices=["1", "2", "3", "4", "5"], default="3")
        except (EOFError, KeyboardInterrupt):
            break

        quality = int(rating)
        sr.review_card(card.word, quality)

        # Feedback
        if quality >= 4:
            console.print("[green]✓ Great! Card strengthened![/green]")
        elif quality >= 3:
            console.print("[yellow]→ Good, keep practicing![/yellow]")
        else:
            console.print("[red]↻ No problem! It'll come back for review soon.[/red]")

    # Summary
    stats = sr.get_stats()
    console.print(f"\n[bold cyan]Review complete![/bold cyan]")
    console.print(f"  Reviewed: {reviewed} cards")
    console.print(f"  Total cards: {stats['total']}")
    console.print(f"  Learned (21d+): {stats['learned']}")
    console.print(f"  Still due: {stats['due']}")


async def quiz_session(sr: SpacedRepetition, question_count: int = 10):
    """Quick vocabulary quiz — see definition, type the word."""
    cards = sr.get_all_cards()
    if len(cards) < 3:
        console.print("[yellow]Add more vocabulary first! Use /read to find new words.[/yellow]")
        return

    selected = random.sample(cards, min(question_count, len(cards)))
    score = 0

    console.print(f"\n[bold cyan]📝 Vocabulary Quiz — {len(selected)} questions[/bold cyan]")
    console.print("[dim]I'll show you the definition — you type the word![/dim]\n")

    for i, card in enumerate(selected, 1):
        # Show definition, hide word
        console.print(f"\n[bold]Q{i}:[/bold] {card.definition}")
        console.print(f"[dim]Context hint: {card.context[:60]}[/dim]")

        try:
            answer = Prompt.ask("Your answer").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        correct = card.word.lower()
        if answer == correct:
            console.print("[green]✓ Correct![/green]")
            score += 1
            sr.review_card(card.word, 5)
        elif answer and (answer in correct or correct in answer):
            console.print(f"[yellow]Almost! The answer is: {card.word}[/yellow]")
            sr.review_card(card.word, 3)
        else:
            console.print(f"[red]✗ The answer is: {card.word}[/red]")
            sr.review_card(card.word, 1)

    console.print(f"\n[bold]Final score: {score}/{len(selected)} ({score * 100 // max(1, len(selected))}%)[/bold]")

    if score == len(selected):
        console.print("[green]🎉 Perfect score! Excellent![/green]")
    elif score >= len(selected) * 0.7:
        console.print("[yellow]Good job! Keep practicing the ones you missed.[/yellow]")
    else:
        console.print("[red]Keep reviewing! Use /review to strengthen these words.[/red]")
