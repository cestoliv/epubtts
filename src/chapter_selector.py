import sys
import inquirer
from rich.console import Console
from rich.table import Table
from rich import box

from .constants import DEFAULT_WPM
from .progress_manager import save_progress

console = Console()

def interactive_chapter_selection(chapters, progress_file, existing_selection=None):
    """Allow user to interactively select which chapters to process."""
    # Use existing selection if available, otherwise default to all chapters
    selected_chapters = set(existing_selection) if existing_selection else set(range(len(chapters)))

    if existing_selection:
        console.print("\n[yellow]📂 Loaded previous chapter selection from progress file[/yellow]")

    # Create choices for inquirer
    choices = []
    for i, chapter in enumerate(chapters):
        word_count = len(chapter['content'].split())
        duration = word_count / DEFAULT_WPM

        # Create a nice display format
        title = chapter['title']
        if len(title) > 60:
            title = title[:57] + "..."

        choice_text = f"{i+1:2d}. {title} ({word_count:,} words, {duration:.1f}m)"
        choices.append((choice_text, i))

    # Calculate stats for current selection
    selected_count = len(selected_chapters)
    total_words = sum(len(chapters[i]['content'].split()) for i in selected_chapters)
    total_duration = total_words / DEFAULT_WPM

    # Show current selection stats
    console.print("\n[bold cyan]📚 Chapter Selection[/bold cyan]")
    console.print(f"[dim]Current selection: {selected_count}/{len(chapters)} chapters | {total_words:,} words | {total_duration:.1f}m[/dim]")

    # Get default selection (currently selected chapters)
    default_selection = [idx for choice_text, idx in choices if idx in selected_chapters]

    # Use inquirer checkbox
    questions = [
        inquirer.Checkbox(
            'chapters',
            message="Select chapters to process (Space=toggle, Enter=confirm):",
            choices=choices,
            default=default_selection,
        ),
    ]

    try:
        answers = inquirer.prompt(questions)
        if answers is None:  # User cancelled (Ctrl+C)
            console.print("[yellow]👋 Selection cancelled. Exiting...[/yellow]")
            sys.exit(0)

        selected_indices = set(answers['chapters'])

        if not selected_indices:
            console.print("[red]❌ No chapters selected! Please select at least one chapter.[/red]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Selection cancelled. Exiting...[/yellow]")
        sys.exit(0)

    # Save chapter selection in progress file (even if no processing progress yet)
    save_progress(progress_file, 0, 0, 0, selected_chapters=sorted(list(selected_indices)))

    console.print("\n[green]💾 Chapter selection saved to progress file[/green]")

    # Return filtered chapters list
    filtered_chapters = [chapters[i] for i in sorted(selected_indices)]

    # Display final selection
    console.print(f"\n[bold green]🎯 Will process {len(filtered_chapters)} chapters:[/bold green]")
    final_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
    final_table.add_column("No.", style="bold", width=6, justify="right")
    final_table.add_column("Title", style="bold")
    final_table.add_column("Words", style="bold", width=10, justify="right")

    for i, chapter in enumerate(filtered_chapters, 1):
        word_count = len(chapter['content'].split())
        final_table.add_row(
            str(i),
            chapter['title'],
            f"{word_count:,}"
        )

    console.print(final_table)

    return filtered_chapters, selected_indices
