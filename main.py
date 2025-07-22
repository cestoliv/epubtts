#!/usr/bin/env python3

import argparse
import sys
import os

from src.audio_generator import convert_text_to_audio
from src.constants import DEFAULT_CHUNK_WORDS, DEFAULT_VOICE


def print_usage():
    print(f"""
Usage: python main.py <input_file> [<output_audio_file>] [options]

Commands:
    -h, --help         Show this help message

Options:
    --voice <str>      Voice to use (default: {DEFAULT_VOICE})
    --hf-repo <str>    HuggingFace repository for models
    --voice-repo <str> HuggingFace repository for voices
    --quantize <int>   Quantization bits (e.g., 8 for 8-bit)
    --reselect-chapters Force interactive chapter selection
    --device <str>     Device to use (cuda/cpu, default: auto-detect)
    --max-chunk-words <int>  Maximum words per chunk (default: {DEFAULT_CHUNK_WORDS})

Input formats:
    .epub              EPUB book input (will process chapters)

Examples:
    python main.py input.epub output.wav
    python main.py input.epub --voice {DEFAULT_VOICE}
    python main.py input.epub --reselect-chapters
    python main.py input.epub --max-chunk-words 750
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Convert epub to speech using Kyutai TTS", add_help=False
    )
    parser.add_argument("input", help="Input EPUB file")
    parser.add_argument("output", nargs="?", help="Output audio file")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice to use for TTS")
    parser.add_argument("--hf-repo", help="HuggingFace repository for models")
    parser.add_argument("--voice-repo", help="HuggingFace repository for voices")
    parser.add_argument("--quantize", type=int, help="Quantization bits")
    parser.add_argument(
        "--reselect-chapters",
        action="store_true",
        help="Force interactive chapter selection even if previous selection exists",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--max-chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help=f"Maximum words per chunk (default: {DEFAULT_CHUNK_WORDS})",
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")

    args = parser.parse_args()

    if args.help:
        print_usage()
        return

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    if not args.input.endswith(".epub"):
        print("Error: Only EPUB files are supported")
        sys.exit(1)

    # Set default output file if not specified
    if not args.output:
        args.output = f"{os.path.splitext(args.input)[0]}.wav"

    # Convert text to audio
    convert_text_to_audio(
        args.input,
        args.output,
        voice=args.voice,
        hf_repo=args.hf_repo,
        voice_repo=args.voice_repo,
        quantize=args.quantize,
        reselect_chapters=args.reselect_chapters,
        device=args.device,
        max_chunk_words=args.max_chunk_words,
    )


if __name__ == "__main__":
    main()
