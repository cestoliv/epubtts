import platform
import json
import queue
import numpy as np
import sys


class MLXBackend:
    """MLX backend for macOS."""

    def __init__(self):
        self.tts_model = None
        self.cfg_coef_conditioning = None
        self.cfg_is_no_prefix = True
        self.cfg_is_no_text = True

    def load_model(self, hf_repo, voice_repo, quantize=None, device="auto"):
        """Load the MLX TTS model."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import sentencepiece
            from moshi_mlx import models
            from moshi_mlx.models.tts import (
                DEFAULT_DSM_TTS_REPO,
                DEFAULT_DSM_TTS_VOICE_REPO,
                TTSModel,
            )
            from moshi_mlx.utils.loaders import hf_get

            mx.random.seed(299792458)

            # Use package defaults if not specified
            if hf_repo is None:
                hf_repo = DEFAULT_DSM_TTS_REPO
            if voice_repo is None:
                voice_repo = DEFAULT_DSM_TTS_VOICE_REPO

            raw_config = hf_get("config.json", hf_repo)
            with open(hf_get(raw_config), "r") as fobj:
                raw_config = json.load(fobj)

            mimi_weights = hf_get(raw_config["mimi_name"], hf_repo)
            moshi_name = raw_config.get("moshi_name", "model.safetensors")
            moshi_weights = hf_get(moshi_name, hf_repo)
            tokenizer = hf_get(raw_config["tokenizer_name"], hf_repo)
            lm_config = models.LmConfig.from_config_dict(raw_config)
            model = models.Lm(lm_config)
            model.set_dtype(mx.bfloat16)

            model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

            if quantize is not None:
                nn.quantize(model.depformer, bits=quantize)
                for layer in model.transformer.layers:
                    nn.quantize(layer.self_attn, bits=quantize)
                    nn.quantize(layer.gating, bits=quantize)

            text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))

            generated_codebooks = lm_config.generated_codebooks
            audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
            audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

            self.tts_model = TTSModel(
                model,
                audio_tokenizer,
                text_tokenizer,
                voice_repo=voice_repo,
                temp=0.6,
                cfg_coef=1,
                max_padding=8,
                initial_padding=2,
                final_padding=2,
                padding_bonus=0,
                raw_config=raw_config,
            )

            if self.tts_model.valid_cfg_conditionings:
                self.cfg_coef_conditioning = self.tts_model.cfg_coef
                self.tts_model.cfg_coef = 1.0
                self.cfg_is_no_text = False
                self.cfg_is_no_prefix = False
            else:
                self.cfg_is_no_text = True
                self.cfg_is_no_prefix = True

        except Exception as e:
            print(f"Error loading MLX TTS model: {e}")
            sys.exit(1)

    def process_chunk(self, chunk, voice):
        """Process a single chunk of text with MLX model."""

        # Prepare the text for TTS
        all_entries = [self.tts_model.prepare_script([chunk])]

        # Get voices
        if self.tts_model.multi_speaker:
            voices = [self.tts_model.get_voice_path(voice)]
        else:
            voices = []

        all_attributes = [
            self.tts_model.make_condition_attributes(voices, self.cfg_coef_conditioning)
        ]

        wav_frames = queue.Queue()

        def _on_frame(frame):
            if (frame == -1).any():
                return
            import mlx.core as mx

            _pcm = self.tts_model.mimi.decode_step(frame[:, :, None])
            _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1))
            wav_frames.put_nowait(_pcm)

        # Generate audio with frame callback
        result = self.tts_model.generate(
            all_entries,
            all_attributes,
            cfg_is_no_prefix=self.cfg_is_no_prefix,
            cfg_is_no_text=self.cfg_is_no_text,
            on_frame=_on_frame,
        )

        # Collect all frames
        frames = []
        while True:
            try:
                frames.append(wav_frames.get_nowait())
            except queue.Empty:
                break

        # Concatenate frames
        if frames:
            wav = np.concat(frames, -1)
            return wav
        else:
            return np.array([])

    @property
    def sample_rate(self):
        """Get the sample rate of the MLX model."""
        return self.tts_model.mimi.sample_rate if self.tts_model else 24000


class PyTorchBackend:
    """PyTorch backend for Windows/Linux."""

    def __init__(self):
        self.tts_model = None

    def __del__(self):
        """Cleanup method to ensure proper resource deallocation."""
        try:
            if hasattr(self, 'tts_model') and self.tts_model is not None:
                self.tts_model = None
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except (ImportError, AttributeError):
            pass  # Ignore errors during cleanup

    def load_model(self, hf_repo, voice_repo, quantize=None, device="auto"):
        """Load the PyTorch TTS model."""
        import torch
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import (
            DEFAULT_DSM_TTS_REPO,
            DEFAULT_DSM_TTS_VOICE_REPO,
            TTSModel,
        )

        # Use package defaults if not specified
        if hf_repo is None:
            hf_repo = DEFAULT_DSM_TTS_REPO
        if voice_repo is None:
            voice_repo = DEFAULT_DSM_TTS_VOICE_REPO

        # Auto-detect device if not specified
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        checkpoint_info = CheckpointInfo.from_hf_repo(hf_repo)
        self.tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=device
        )

    def process_chunk(self, chunk, voice):
        """Process a single chunk of text with PyTorch model."""

        import torch
        import gc

        try:
            # Prepare the text for TTS
            entries = self.tts_model.prepare_script([chunk], padding_between=1)
            voice_path = self.tts_model.get_voice_path(voice)

            # CFG coef goes here because the model was trained with CFG distillation,
            # so it's not _actually_ doing CFG at inference time.
            condition_attributes = self.tts_model.make_condition_attributes(
                [voice_path], cfg_coef=2.0
            )

            # Generate audio
            result = self.tts_model.generate([entries], [condition_attributes])

            # Decode frames with proper memory management
            pcms = []
            try:
                with self.tts_model.mimi.streaming(1), torch.no_grad():
                    for frame in result.frames[self.tts_model.delay_steps :]:
                        # Process frame and immediately move to CPU/numpy
                        pcm_tensor = self.tts_model.mimi.decode(frame[:, 1:, :])
                        pcm_numpy = np.clip(pcm_tensor.cpu().numpy()[0, 0], -1, 1)
                        pcms.append(pcm_numpy)
                        
                        # Clean up GPU memory for this frame
                        del pcm_tensor
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Concatenate results
                if pcms:
                    audio_result = np.concatenate(pcms, axis=-1)
                else:
                    audio_result = np.array([])
                    
            finally:
                # Cleanup PCM list and result
                pcms.clear()
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            return audio_result
            
        except Exception as e:
            # Ensure cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e

    @property
    def sample_rate(self):
        """Get the sample rate of the PyTorch model."""
        return self.tts_model.mimi.sample_rate if self.tts_model else 24000


class TTSBackend:
    """TTS backend that creates fresh instances for each chunk."""

    def __init__(self, backend_type="auto"):
        self.backend_type = backend_type
        self.hf_repo = None
        self.voice_repo = None
        self.quantize = None
        self.device = "auto"
        self._sample_rate = 24000

    def load_model(self, hf_repo, voice_repo, quantize=None, device="auto"):
        """Store model parameters for creating fresh backends."""
        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.quantize = quantize
        self.device = device

        # Get sample rate from a test backend
        test_backend = self._create_backend()
        self._sample_rate = test_backend.sample_rate
        del test_backend

    def _create_backend(self):
        """Create a fresh backend instance."""
        # Auto-detect platform if needed
        if self.backend_type == "auto":
            if platform.system() == "Darwin":  # macOS
                try:
                    import importlib.util

                    if importlib.util.find_spec("mlx.core") is not None:
                        backend = MLXBackend()
                    else:
                        raise ImportError()
                except ImportError:
                    print("MLX not available, using PyTorch")
                    backend = PyTorchBackend()
            else:
                backend = PyTorchBackend()
        elif self.backend_type == "mlx":
            backend = MLXBackend()
        elif self.backend_type == "pytorch":
            backend = PyTorchBackend()
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

        backend.load_model(self.hf_repo, self.voice_repo, self.quantize, self.device)
        return backend

    def process_chunk(self, chunk, voice):
        """Process chunk with a fresh backend instance."""
        import gc
        
        backend = self._create_backend()
        try:
            return backend.process_chunk(chunk, voice)
        finally:
            # Explicit cleanup of backend
            if hasattr(backend, 'tts_model') and backend.tts_model is not None:
                # Clear model references
                backend.tts_model = None
            del backend
            
            # Force garbage collection and GPU memory cleanup
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # PyTorch not available (MLX backend)

    @property
    def sample_rate(self):
        return self._sample_rate


def get_backend(backend_type="auto"):
    """Get TTS backend."""
    return TTSBackend(backend_type)
