import os
import sys
import threading
import signal
from rich.console import Console
from rich.table import Table
from rich import box
import time

from .constants import DEFAULT_CHUNK_WORDS, DEFAULT_VOICE
from .epub_parser import extract_chapters_from_epub
from .text_processing import chunk_text
from .chapter_selector import interactive_chapter_selection
from .progress_manager import load_progress, save_progress
from .audio_utils import append_audio_to_wav
from .tts_backend import get_backend

console = Console()


def handle_ctrl_c(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nCtrl+C detected, stopping...")
    spinner_manager.stop_spinner = True
    spinner_manager.stop_audio = True
    sys.exit(0)


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_ctrl_c)


class SpinnerManager:
    """Simple spinner manager using Rich."""

    def __init__(self):
        self.stop_spinner = False
        self.stop_audio = False

    def spinning_wheel(self, message="Processing...", progress=None):
        """Display a rich spinner with message."""
        with console.status(f"[bold green]{message}") as status:
            while not self.stop_spinner:
                if progress:
                    status.update(f"[bold green]{message} {progress}")
                time.sleep(0.1)


spinner_manager = SpinnerManager()


def convert_text_to_audio(
    input_file,
    output_file=None,
    voice=DEFAULT_VOICE,
    hf_repo=None,
    voice_repo=None,
    quantize=None,
    reselect_chapters=False,
    device="auto",
    max_chunk_words=DEFAULT_CHUNK_WORDS,
):
    """Convert EPUB text to audio using Kyutai TTS."""

    # Create TTS backend (always uses fresh process for each chunk)
    backend = get_backend(backend_type="auto")
    backend.load_model(hf_repo, voice_repo, quantize, device)

    # Read the EPUB file
    chapters = extract_chapters_from_epub(input_file)
    if not chapters:
        print("No chapters found in EPUB file.")
        sys.exit(1)

    sample_rate = backend.sample_rate

    # Set default output file if not specified
    if not output_file:
        output_file = f"{os.path.splitext(input_file)[0]}.wav"

    progress_file = f"{output_file}.progress"
    existing_progress = load_progress(progress_file)
    existing_selection = (
        existing_progress.get("selected_chapters") if existing_progress else None
    )

    # Chapter selection logic
    if reselect_chapters or not existing_selection:
        chapters, selected_indices = interactive_chapter_selection(
            chapters, progress_file, existing_selection
        )
    else:
        # Load existing selection from progress file
        selected_indices = set(existing_selection)
        chapters = [chapters[i] for i in sorted(selected_indices)]
        console.print(
            f"\n[green]📋 Using previous chapter selection: {len(chapters)} chapters selected[/green]"
        )

        # Create a table for the selected chapters
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
        table.add_column("No.", style="bold", width=6, justify="right")
        table.add_column("Title", style="bold")
        table.add_column("Words", style="bold", width=10, justify="right")

        for i, chapter in enumerate(chapters, 1):
            word_count = len(chapter["content"].split())
            table.add_row(str(i), chapter["title"], f"{word_count:,}")

        console.print(table)

    # Process chapters
    progress = load_progress(progress_file)
    start_chapter = 0
    start_chunk = 0

    if progress:
        start_chapter = progress["chapter_index"]
        start_chunk = progress["chunk_index"]
        console.print(
            f"\n[yellow]🔄 Resuming from chapter {start_chapter + 1}, chunk {start_chunk + 1}[/yellow]"
        )
        console.print(
            f"[dim]Will start processing from chapter index {start_chapter}: '{chapters[start_chapter]['title']}'[/dim]"
        )
    else:
        console.print(f"\n[blue]🎬 Starting fresh processing to {output_file}[/blue]")

    total_processed = 0

    for chapter_num, chapter in enumerate(chapters[start_chapter:], start_chapter):
        if spinner_manager.stop_audio:
            break

        console.print(f"\n[bold blue]🎧 Processing: {chapter['title']}[/bold blue]")

        # Use improved chunking that respects sentence boundaries
        chunks = chunk_text(chapter["content"], max_chunk_words=max_chunk_words)
        processed_chunks = 0
        total_chunks = len(chunks)

        console.print(
            f"[dim]{total_chunks} chunks (max {max_chunk_words} words per chunk)[/dim]"
        )

        # Skip chunks that were already processed
        chunk_start = start_chunk if chapter_num == start_chapter else 0

        if chapter_num == start_chapter and chunk_start > 0:
            console.print(
                f"Skipping first {chunk_start} chunks in chapter {chapter_num}"
            )

        for chunk_num, chunk in enumerate(chunks[chunk_start:], chunk_start + 1):
            if spinner_manager.stop_audio:
                break

            # Start spinner in background thread
            spinner_manager.stop_spinner = False
            spinner_thread = threading.Thread(
                target=spinner_manager.spinning_wheel,
                args=(f"Processing chunk {chunk_num}/{total_chunks}",),
            )
            spinner_thread.start()

            try:
                # Process chunk with backend
                audio_data = backend.process_chunk(chunk, voice)

                # Append to WAV file incrementally
                is_first_chunk = chapter_num == 0 and chunk_num == 1 and not progress
                append_audio_to_wav(
                    output_file, audio_data, sample_rate, is_first_chunk
                )

                processed_chunks += 1
                total_processed += 1

                # Save progress after each chunk
                save_progress(
                    progress_file,
                    chapter_num,
                    chunk_num,
                    total_chunks,
                    selected_chapters=sorted(list(selected_indices)),
                )

            except Exception as e:
                print(f"\nError processing chunk {chunk_num}: {e}")

            # Stop spinner
            spinner_manager.stop_spinner = True
            spinner_thread.join()

        console.print(
            f"\n[green]✅ Completed {chapter['title']}: {processed_chunks}/{total_chunks} chunks processed[/green]"
        )

        # Reset start_chunk for next chapter
        start_chunk = 0

    # Clean up progress file when complete
    if not spinner_manager.stop_audio and os.path.exists(progress_file):
        os.remove(progress_file)

    console.print(
        f"\n[bold green]🎉 Created {output_file} with {total_processed} chunks processed[/bold green]"
    )
