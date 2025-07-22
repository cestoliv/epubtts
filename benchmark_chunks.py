#!/usr/bin/env python3
"""
Benchmark script to detect maximum chunk size and generation time for TTS.

This script will test progressively larger chunk sizes until memory limits
are reached or the system becomes unstable.
"""

import time
import gc
import sys
import platform
import psutil
import numpy as np
from pathlib import Path

from src.tts_backend import get_backend
from src.text_processing import chunk_text
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import box

console = Console()


def get_system_info():
    """Get system information for the benchmark."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": psutil.cpu_count(),
        "total_memory": psutil.virtual_memory().total / (1024**3),  # GB
        "python_version": sys.version.split()[0],
    }

    # Check for GPU - try MLX first (Apple Silicon), then PyTorch
    try:
        import mlx.core
        info["gpu"] = "Apple Silicon (MLX)"
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            else:
                info["gpu"] = "CPU only"
        except ImportError:
            info["gpu"] = "Unknown"

    return info


def generate_test_text(word_count):
    """Generate test text with specified word count."""
    # Use a varied vocabulary to better simulate real text
    words = [
        "The",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
        "This",
        "is",
        "a",
        "comprehensive",
        "test",
        "of",
        "text-to-speech",
        "generation",
        "with",
        "various",
        "sentence",
        "structures",
        "and",
        "vocabulary",
        "to",
        "simulate",
        "real",
        "world",
        "content",
        "processing",
        "for",
        "audiobook",
        "creation",
        "We",
        "need",
        "to",
        "ensure",
        "that",
        "our",
        "system",
        "can",
        "handle",
        "different",
        "types",
        "of",
        "content",
        "including",
        "technical",
        "terms",
        "narrative",
        "prose",
        "dialogue",
        "and",
        "descriptive",
        "passages",
    ]

    # Create text with proper sentence structure
    text_parts = []
    current_words = 0

    while current_words < word_count:
        # Create sentences of 8-15 words
        sentence_length = min(np.random.randint(8, 16), word_count - current_words)
        sentence_words = np.random.choice(words, sentence_length, replace=True)
        sentence = " ".join(sentence_words) + "."
        text_parts.append(sentence)
        current_words += sentence_length

    return " ".join(text_parts)


def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # MB


def benchmark_chunk_size(chunk_words, voice, max_attempts=3):
    """Benchmark a specific chunk size with fresh model instance."""
    results = []

    for attempt in range(max_attempts):
        backend = None
        try:
            # Generate test text
            test_text = generate_test_text(chunk_words)

            # Create fresh backend instance for each test
            backend = get_backend("auto")
            backend.load_model(None, None, None, "auto")

            # Measure memory before processing
            memory_before = measure_memory_usage()

            # Record start time and measure peak memory during processing
            start_time = time.time()
            peak_memory = memory_before

            # Process the chunk with memory monitoring
            class MemoryMonitor:
                def __init__(self):
                    self.peak = memory_before
                
                def update(self):
                    current = measure_memory_usage()
                    if current > self.peak:
                        self.peak = current

            monitor = MemoryMonitor()
            
            # Start processing
            audio_data = backend.process_chunk(test_text, voice)
            
            # Update memory peak one final time
            monitor.update()
            peak_memory = monitor.peak

            # Record end time
            end_time = time.time()

            # Calculate metrics
            generation_time = end_time - start_time
            memory_used = peak_memory - memory_before
            audio_duration = (
                len(audio_data) / backend.sample_rate if len(audio_data) > 0 else 0
            )
            realtime_factor = (
                audio_duration / generation_time if generation_time > 0 else 0
            )

            results.append(
                {
                    "generation_time": generation_time,
                    "memory_used": memory_used,
                    "audio_duration": audio_duration,
                    "realtime_factor": realtime_factor,
                    "success": True,
                    "audio_samples": len(audio_data),
                }
            )

        except Exception as e:
            results.append(
                {
                    "generation_time": 0,
                    "memory_used": 0,
                    "audio_duration": 0,
                    "realtime_factor": 0,
                    "success": False,
                    "error": str(e),
                    "audio_samples": 0,
                }
            )
        finally:
            # Always cleanup backend and memory
            if backend is not None:
                # Explicitly delete backend to unload model
                del backend
            
            # Force garbage collection
            gc.collect()
            
            # GPU memory cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # MLX memory cleanup if available
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

    return results


def run_benchmark():
    """Run the complete benchmark suite."""
    console.print("[bold blue]🚀 TTS Chunk Size Benchmark[/bold blue]\n")

    # Get system info
    sys_info = get_system_info()

    # Display system info
    info_table = Table(title="System Information", box=box.SIMPLE)
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")

    for key, value in sys_info.items():
        if isinstance(value, float):
            info_table.add_row(key.replace("_", " ").title(), f"{value:.1f}")
        else:
            info_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(info_table)
    console.print()

    # Set default voice for testing
    voice = "unmute-prod-website/fabieng-enhanced-v2.wav"
    console.print("[green]✅ Using fresh model instances for each test[/green]\n")

    # Define test chunk sizes (in words)
    test_sizes = [
        50,
        100,
        200,
        300,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        5000,
        7500,
        10000,
    ]

    # Results storage
    benchmark_results = []

    # Run benchmark
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for chunk_size in test_sizes:
            task = progress.add_task(f"Testing {chunk_size} words...", total=1)

            console.print(
                f"\n[bold cyan]📊 Testing chunk size: {chunk_size} words[/bold cyan]"
            )

            # Run benchmark for this chunk size
            results = benchmark_chunk_size(chunk_size, voice)

            # Analyze results
            successful_runs = [r for r in results if r["success"]]

            if successful_runs:
                avg_time = np.mean([r["generation_time"] for r in successful_runs])
                avg_memory = np.mean([r["memory_used"] for r in successful_runs])
                avg_duration = np.mean([r["audio_duration"] for r in successful_runs])
                avg_rtf = np.mean([r["realtime_factor"] for r in successful_runs])

                benchmark_results.append(
                    {
                        "chunk_size": chunk_size,
                        "success_rate": len(successful_runs) / len(results),
                        "avg_generation_time": avg_time,
                        "avg_memory_used": avg_memory,
                        "avg_audio_duration": avg_duration,
                        "avg_realtime_factor": avg_rtf,
                        "failed": len(results) - len(successful_runs),
                    }
                )


                console.print(
                    f"  ✅ Success rate: {len(successful_runs)}/{len(results)}"
                )
                console.print(f"  ⏱️  Avg generation time: {avg_time:.2f}s")
                console.print(f"  🎵 Avg audio duration: {avg_duration:.2f}s")
                console.print(f"  ⚡ Avg realtime factor: {avg_rtf:.2f}x")
                console.print(f"  💾 Avg memory used: {avg_memory:.1f} MB")

            else:
                # All runs failed
                error_msgs = [
                    r.get("error", "Unknown error") for r in results if not r["success"]
                ]
                console.print("  ❌ All runs failed")
                console.print(f"  🔍 Common errors: {', '.join(set(error_msgs))}")

                benchmark_results.append(
                    {
                        "chunk_size": chunk_size,
                        "success_rate": 0,
                        "avg_generation_time": 0,
                        "avg_memory_used": 0,
                        "avg_audio_duration": 0,
                        "avg_realtime_factor": 0,
                        "failed": len(results),
                        "errors": error_msgs,
                    }
                )

                # If we hit 2 consecutive failures, stop testing larger sizes
                if (
                    len(benchmark_results) >= 2
                    and benchmark_results[-1]["success_rate"] == 0
                    and benchmark_results[-2]["success_rate"] == 0
                ):
                    console.print(
                        "[yellow]⚠️ Stopping benchmark after consecutive failures[/yellow]"
                    )
                    break

            progress.update(task, advance=1)

    # Display final results
    console.print("\n[bold green]📋 Benchmark Results Summary[/bold green]\n")

    results_table = Table(title="Chunk Size Performance", box=box.ROUNDED)
    results_table.add_column("Words", justify="right", style="bold")
    results_table.add_column("Success", justify="center")
    results_table.add_column("Gen Time", justify="right")
    results_table.add_column("Audio Dur", justify="right")
    results_table.add_column("RTF", justify="right")
    results_table.add_column("Memory", justify="right")

    for result in benchmark_results:
        if result["success_rate"] > 0:
            success_color = "green" if result["success_rate"] == 1.0 else "yellow"
            results_table.add_row(
                str(result["chunk_size"]),
                f"[{success_color}]{result['success_rate']:.0%}[/{success_color}]",
                f"{result['avg_generation_time']:.2f}s",
                f"{result['avg_audio_duration']:.2f}s",
                f"{result['avg_realtime_factor']:.2f}x",
                f"{result['avg_memory_used']:.0f}MB",
            )
        else:
            results_table.add_row(
                str(result["chunk_size"]),
                "[red]FAIL[/red]",
                "[dim]--[/dim]",
                "[dim]--[/dim]",
                "[dim]--[/dim]",
                "[dim]--[/dim]",
            )

    console.print(results_table)


    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"

    with open(results_file, "w") as f:
        f.write("TTS Chunk Size Benchmark Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("System Information:\n")
        for key, value in sys_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBenchmark Results:\n")
        for result in benchmark_results:
            f.write(f"Chunk Size: {result['chunk_size']} words\n")
            f.write(f"Success Rate: {result['success_rate']:.1%}\n")
            if result["success_rate"] > 0:
                f.write(f"Avg Generation Time: {result['avg_generation_time']:.2f}s\n")
                f.write(f"Avg Audio Duration: {result['avg_audio_duration']:.2f}s\n")
                f.write(f"Avg Realtime Factor: {result['avg_realtime_factor']:.2f}x\n")
                f.write(f"Avg Memory Used: {result['avg_memory_used']:.1f}MB\n")
            f.write("-" * 30 + "\n")

    console.print(f"\n💾 Results saved to: [bold blue]{results_file}[/bold blue]")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Benchmark cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Benchmark failed: {e}[/red]")
        raise
