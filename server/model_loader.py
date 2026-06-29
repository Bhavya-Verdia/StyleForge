"""
model_loader.py — Thread-safe singleton for loading base + fine-tuned models using MLX.

Environment variables:
    BASE_MODEL_ID   Mistral model on HuggingFace Hub (default: mlx-community/Mistral-7B-Instruct-v0.3-4bit)
    ADAPTER_DIR     Path to saved QLoRA adapter weights
    HF_TOKEN        HuggingFace API token (for gated models)
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
    from mlx_lm import load
except ImportError:
    load = None  # type: ignore

BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
ADAPTER_DIR: str   = os.getenv("ADAPTER_DIR",   "./adapter_weights")
HF_TOKEN: str | None = os.getenv("HF_TOKEN", None)


class ModelLoader:
    """Loads and caches base and fine-tuned models using MLX."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None

    # ── Public API ────────────────────────────────────────────

    def load_base_model(self) -> None:
        logger.info("[ModelLoader] Loading 4-bit base model: %s…", BASE_MODEL_ID)
        self.base_model, self.base_tokenizer = load(BASE_MODEL_ID)
        logger.info("[ModelLoader] Base model ready.")

    def load_finetuned_model(self, adapter_path: str = None) -> None:
        if adapter_path is None:
            adapter_path = ADAPTER_DIR

        if not os.path.exists(adapter_path):
            logger.warning("[ModelLoader] Adapter directory not found: %s", adapter_path)
            logger.warning("[ModelLoader] Fine-tuned endpoint will return 503 until adapters are present.")
            return

        with self._lock:
            # Clear previous model to free VRAM
            if self.finetuned_model is not None:
                del self.finetuned_model
                del self.finetuned_tokenizer
                self.finetuned_model = None
                import mlx.core as mx
                mx.metal.clear_cache()

            logger.info("[ModelLoader] Loading fine-tuned model with adapter: %s", adapter_path)
            self.finetuned_model, self.finetuned_tokenizer = load(BASE_MODEL_ID, adapter_path=adapter_path)
            logger.info("[ModelLoader] Fine-tuned model ready.")

    def load_all(self) -> None:
        with self._lock:
            if load is None:
                raise RuntimeError(
                    "mlx-lm is not installed. Run `pip install mlx-lm`.\n"
                    "Set USE_MOCK=true to run the server without loading models."
                )
            self.load_base_model()
            self.load_finetuned_model()
            logger.info("[ModelLoader] All MLX models loaded successfully.")

    def cleanup(self) -> None:
        """Deallocate models."""
        with self._lock:
            for attr in ("base_model", "finetuned_model", "base_tokenizer", "finetuned_tokenizer"):
                if getattr(self, attr, None) is not None:
                    setattr(self, attr, None)
            logger.info("[ModelLoader] Models unloaded.")
