import sys
import os
from rich.console import Console
from rich.table import Table
from rich import box

from .constants import DEFAULT_WPM
from .progress_manager import save_progress

console = Console()


def is_interactive_supported():
    """Check if the environment supports interactive prompts."""
    # Check for common non-interactive environments
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False

    # Check for Google Colab specifically
    try:
        import google.colab

        return False
    except ImportError:
        pass

    # Check for Jupyter environment
    if "JUPYTER_RUNTIME_DIR" in os.environ or "JPY_PARENT_PID" in os.environ:
        return False

    return True


def simple_chapter_selection(chapters, progress_file, existing_selection=None):
    """Simple chapter selection for non-interactive environments like Google Colab."""
    # Use existing selection if available, otherwise default to all chapters
    selected_chapters = (
        set(existing_selection) if existing_selection else set(range(len(chapters)))
    )

    if existing_selection:
        console.print(
            "\n[yellow]📂 Loaded previous chapter selection from progress file[/yellow]"
        )

    # Show available chapters
    console.print("\n[bold cyan]📚 Available Chapters:[/bold cyan]")

    selection_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    selection_table.add_column("No.", style="bold", width=6, justify="right")
    selection_table.add_column("Title", style="bold")
    selection_table.add_column("Words", style="bold", width=10, justify="right")
    selection_table.add_column("Duration", style="bold", width=10, justify="right")

    total_words = 0
    for i, chapter in enumerate(chapters):
        word_count = len(chapter["content"].split())
        total_words += word_count
        duration = word_count / DEFAULT_WPM

        title = chapter["title"]
        if len(title) > 50:
            title = title[:47] + "..."

        # Mark currently selected chapters
        marker = "✓" if i in selected_chapters else " "
        selection_table.add_row(
            f"{marker} {i + 1}", title, f"{word_count:,}", f"{duration:.1f}m"
        )

    console.print(selection_table)

    # Show current selection stats
    if existing_selection:
        selected_words = sum(
            len(chapters[i]["content"].split()) for i in selected_chapters
        )
        selected_duration = selected_words / DEFAULT_WPM
        console.print(
            f"\n[dim]Current selection: {len(selected_chapters)}/{len(chapters)} chapters | {selected_words:,} words | {selected_duration:.1f}m[/dim]"
        )

    # Get user input
    console.print(f"\n[bold yellow]Chapter Selection Instructions:[/bold yellow]")
    console.print("• Enter chapter numbers separated by commas (e.g., 1,3,5-7,10)")
    console.print("• Use ranges with dashes (e.g., 1-3 means chapters 1, 2, and 3)")
    console.print("• Leave empty to select ALL chapters")

    try:
        user_input = input("\nEnter your selection: ").strip()

        if not user_input:
            # Empty input = select all chapters
            selected_indices = set(range(len(chapters)))
            console.print("[green]📋 Selected all chapters[/green]")
        else:
            # Parse the input
            selected_indices = set()
            for part in user_input.split(","):
                part = part.strip()
                if "-" in part:
                    # Handle ranges like "1-3"
                    try:
                        start, end = part.split("-")
                        start_idx = int(start.strip()) - 1  # Convert to 0-based
                        end_idx = int(end.strip()) - 1
                        if (
                            start_idx >= 0
                            and end_idx < len(chapters)
                            and start_idx <= end_idx
                        ):
                            selected_indices.update(range(start_idx, end_idx + 1))
                        else:
                            console.print(f"[red]⚠️ Invalid range: {part}[/red]")
                    except ValueError:
                        console.print(f"[red]⚠️ Invalid range format: {part}[/red]")
                else:
                    # Handle single numbers
                    try:
                        chapter_num = int(part.strip())
                        if 1 <= chapter_num <= len(chapters):
                            selected_indices.add(chapter_num - 1)  # Convert to 0-based
                        else:
                            console.print(
                                f"[red]⚠️ Chapter {chapter_num} does not exist (valid range: 1-{len(chapters)})[/red]"
                            )
                    except ValueError:
                        console.print(f"[red]⚠️ Invalid chapter number: {part}[/red]")

            if not selected_indices:
                console.print(
                    "[red]❌ No valid chapters selected! Selecting all chapters.[/red]"
                )
                selected_indices = set(range(len(chapters)))

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Selection cancelled. Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(
            f"[red]❌ Error reading input: {e}. Selecting all chapters.[/red]"
        )
        selected_indices = set(range(len(chapters)))

    # Save chapter selection in progress file
    save_progress(
        progress_file, 0, 0, 0, selected_chapters=sorted(list(selected_indices))
    )
    console.print("\n[green]💾 Chapter selection saved to progress file[/green]")

    # Return filtered chapters list
    filtered_chapters = [chapters[i] for i in sorted(selected_indices)]

    # Display final selection
    console.print(
        f"\n[bold green]🎯 Will process {len(filtered_chapters)} chapters:[/bold green]"
    )
    final_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
    final_table.add_column("No.", style="bold", width=6, justify="right")
    final_table.add_column("Title", style="bold")
    final_table.add_column("Words", style="bold", width=10, justify="right")

    for i, chapter in enumerate(filtered_chapters, 1):
        word_count = len(chapter["content"].split())
        final_table.add_row(str(i), chapter["title"], f"{word_count:,}")

    console.print(final_table)
    return filtered_chapters, selected_indices


def interactive_chapter_selection(chapters, progress_file, existing_selection=None):
    """Allow user to interactively select which chapters to process."""

    # Check if we can use interactive prompts
    if not is_interactive_supported():
        console.print(
            "[yellow]🔄 Interactive selection not supported in this environment (Google Colab/Jupyter). Using simple input mode.[/yellow]"
        )
        return simple_chapter_selection(chapters, progress_file, existing_selection)

    return inquirer_chapter_selection(chapters, progress_file, existing_selection)


def inquirer_chapter_selection(chapters, progress_file, existing_selection=None):
    """Interactive chapter selection using inquirer (for supported terminals)."""
    import inquirer

    # Use existing selection if available, otherwise default to all chapters
    selected_chapters = (
        set(existing_selection) if existing_selection else set(range(len(chapters)))
    )

    if existing_selection:
        console.print(
            "\n[yellow]📂 Loaded previous chapter selection from progress file[/yellow]"
        )

    # Create choices for inquirer
    choices = []
    for i, chapter in enumerate(chapters):
        word_count = len(chapter["content"].split())
        duration = word_count / DEFAULT_WPM

        # Create a nice display format
        title = chapter["title"]
        if len(title) > 60:
            title = title[:57] + "..."

        choice_text = f"{i + 1:2d}. {title} ({word_count:,} words, {duration:.1f}m)"
        choices.append((choice_text, i))

    # Calculate stats for current selection
    selected_count = len(selected_chapters)
    total_words = sum(len(chapters[i]["content"].split()) for i in selected_chapters)
    total_duration = total_words / DEFAULT_WPM

    # Show current selection stats
    console.print("\n[bold cyan]📚 Chapter Selection[/bold cyan]")
    console.print(
        f"[dim]Current selection: {selected_count}/{len(chapters)} chapters | {total_words:,} words | {total_duration:.1f}m[/dim]"
    )

    # Get default selection (currently selected chapters)
    default_selection = [
        idx for choice_text, idx in choices if idx in selected_chapters
    ]

    # Use inquirer checkbox
    questions = [
        inquirer.Checkbox(
            "chapters",
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

        selected_indices = set(answers["chapters"])

        if not selected_indices:
            console.print(
                "[red]❌ No chapters selected! Please select at least one chapter.[/red]"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Selection cancelled. Exiting...[/yellow]")
        sys.exit(0)

    # Save chapter selection in progress file (even if no processing progress yet)
    save_progress(
        progress_file, 0, 0, 0, selected_chapters=sorted(list(selected_indices))
    )

    console.print("\n[green]💾 Chapter selection saved to progress file[/green]")

    # Return filtered chapters list
    filtered_chapters = [chapters[i] for i in sorted(selected_indices)]

    # Display final selection
    console.print(
        f"\n[bold green]🎯 Will process {len(filtered_chapters)} chapters:[/bold green]"
    )
    final_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
    final_table.add_column("No.", style="bold", width=6, justify="right")
    final_table.add_column("Title", style="bold")
    final_table.add_column("Words", style="bold", width=10, justify="right")

    for i, chapter in enumerate(filtered_chapters, 1):
        word_count = len(chapter["content"].split())
        final_table.add_row(str(i), chapter["title"], f"{word_count:,}")

    console.print(final_table)

    return filtered_chapters, selected_indices
