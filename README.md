<img src="assets/logo.png" alt="EpubTTS Logo" width="150">
<br><br>

# EpubTTS

Convert EPUB books to high-quality audio using Kyutai TTS models. Cross-platform support with optimized backends for macOS Apple Silicon (MLX) and Windows/Linux (PyTorch).

**Support every language Kyutai TTS supports! Big mention to english and french voices.**

## ✨ Features

- 📚 **EPUB Processing**: Smart chapter extraction and text processing
- 🎯 **Interactive Selection**: Choose which chapters to convert
- 🔄 **Resume Support**: Continue interrupted conversions seamlessly
- 🌍 **Cross-Platform**: Optimized for macOS, Windows, and Linux
- 🎤 **Multiple Voices**: Access to Kyutai's diverse voice collection (including English and French)
- ⚡ **Smart Chunking**: Word-based chunking with sentence boundary respect
- 🔊 **Audio Normalization**: Consistent volume levels across chunks
- 📊 **Progress Tracking**: Visual indicators and detailed logging

## 🚀 Installation

### Dependencies
**First install [uv](https://github.com/astral-sh/uv)**

### Clone the Repository
```bash
git clone https://git.chevro.fr/cestoliv/epubtts.git
cd epubtts
```

### Platform-Specific Setup

#### 🍎 macOS (Apple Silicon) - Recommended
```bash
uv sync --group mlx
```

#### 🪟 Windows / 🐧 Linux
```bash
uv sync --group pytorch
```

> [!WARNING]
> **⚠️ Important**: `moshi` and `moshi_mlx` packages conflict. Use the correct group for your platform.

## 🎯 Usage

### Basic Conversion
```bash
# Convert entire book
uv run python main.py book.epub

# Specify output filename
uv run python main.py book.epub audiobook.wav
```

### Voice Selection
```bash
# Use specific voice
uv run python main.py book.epub --voice "expresso/ex01-ex02_default_001_channel2_198s.wav"

# Browse available voices at: https://kyutai.org/next/tts
# Use the voice name from the test box (path ending with .wav)
```

### Advanced Options
```bash
# Adjust chunk size (default: 500 words)
uv run python main.py book.epub --max-chunk-words 300

# Force chapter reselection
uv run python main.py book.epub --reselect-chapters

# Specify device (PyTorch only)
uv run python main.py book.epub --device cuda

# Enable quantization (MLX only)
uv run python main.py book.epub --quantize 8
```

### Custom Models
```bash
# Use custom HuggingFace repositories
uv run python main.py book.epub --hf-repo custom/model --voice-repo custom/voices
```

## 🎤 Voice Options

Explore different voices at **[Kyutai TTS Demo](https://kyutai.org/next/tts)**:

1. Visit the demo page
2. Test different voices using the interface
3. Copy the voice name displayed under each test box (ending with `.wav`)
4. Use it with the `--voice` parameter

**Recommended voices:**
- `expresso/ex01-ex02_default_001_channel2_198s.wav` (English female)
- `unmute-prod-website/fabieng-enhanced-v2.wav` (French male, default)

## 🖥️ Platform Support

| Platform | Backend | GPU Support | Performance |
|----------|---------|-------------|-------------|
| macOS (Apple Silicon) | MLX | ✅ Metal | Excellent |
| macOS (Intel) | PyTorch | ❌ CPU only | Good |
| Windows | PyTorch | ✅ CUDA | Excellent |
| Linux | PyTorch | ✅ CUDA | Excellent |

## 🧪 Testing

Run the test suite:
```bash
uv run pytest tests/ -v
```

## 🔧 Troubleshooting

### Common Issues

**Import/Module Errors:**
```bash
# Ensure you're using uv to run
uv run python main.py book.epub
```

**CUDA Issues (Windows/Linux):**
- Install NVIDIA drivers and CUDA toolkit
- Verify with: `nvidia-smi`

**Memory Issues:**
- Use quantization: `--quantize 8` (MLX)
- Reduce chunk size: `--max-chunk-words 200`

**Audio Quality Problems:**
- Try different voices from [Kyutai TTS Demo](https://kyutai.org/next/tts)
- Reduce chunk size to prevent model state degradation

## 🙏 Acknowledgments

Special thanks to the following projects, which inspired this work:
- [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts)
- [Delayed Streams Modeling: Kyutai STT & TTS](https://github.com/kyutai-labs/delayed-streams-modeling)
