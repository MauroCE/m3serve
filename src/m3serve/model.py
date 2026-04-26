"""BGE-M3 encoder: tokenise -> GPU forward pass -> post-process."""

import copy
import importlib.util
import logging
import os
import threading
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from .types import EmbeddingResult

logger = logging.getLogger(__name__)

# Attention implementations understood by transformers AutoModel.
_VALID_ATTN_IMPLS = frozenset({"flash_attention_2", "flash_attention_3", "sdpa", "eager"})


def _flash_attn_available() -> bool:
    """Return True if the flash-attn package is installed."""
    return importlib.util.find_spec("flash_attn") is not None


def _flash_attn3_available() -> bool:
    """Return True if a FA3-capable flash-attn is installed (v3+ or separate flash_attn_3)."""
    if importlib.util.find_spec("flash_attn_3") is not None:
        return True
    if not _flash_attn_available():
        return False
    try:
        return int(_pkg_version("flash-attn").split(".")[0]) >= 3
    except (PackageNotFoundError, ValueError, IndexError):
        return False


def _resolve_attn_implementation(
    device: str,
    dtype: torch.dtype,
    requested: str | None,
) -> str:
    """Choose the best attention implementation for the given device and dtype.

    When *requested* is None, auto-selects the fastest available option and logs
    the choice at INFO level. When *requested* is set explicitly, raises ValueError
    if the hardware or packages cannot satisfy it.
    """
    if requested is not None and requested not in _VALID_ATTN_IMPLS:
        raise ValueError(
            f"Unknown attn_implementation {requested!r}. Valid choices: {sorted(_VALID_ATTN_IMPLS)}"
        )

    auto = requested is None
    fp16_compat = dtype in (torch.float16, torch.bfloat16)

    # FA2/FA3 cannot run in fp32.
    if not fp16_compat:
        if not auto and requested in ("flash_attention_2", "flash_attention_3"):
            raise ValueError(
                f"{requested} requires float16 or bfloat16, got {dtype}. "
                "Set use_fp16=True or choose attn_implementation='sdpa'."
            )
        return requested or "eager"

    if device == "cpu":
        if not auto and requested in ("flash_attention_2", "flash_attention_3"):
            raise ValueError(f"{requested} is not available on CPU.")
        return requested or "sdpa"

    if device == "mps":
        if not auto and requested in ("flash_attention_2", "flash_attention_3"):
            raise ValueError(f"{requested} is not available on MPS.")
        return requested or "sdpa"

    if not device.startswith("cuda"):
        # Unknown accelerator: SDPA is the safest universal fallback.
        return requested or "sdpa"

    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    major, _ = torch.cuda.get_device_capability(device_idx)
    device_name = torch.cuda.get_device_name(device_idx)

    if major < 8:
        # Turing and older: no FA kernel support.
        if not auto and requested in ("flash_attention_2", "flash_attention_3"):
            raise ValueError(
                f"{requested} requires CUDA compute capability >= 8.0 (Ampere). "
                f"Device {device_name!r} is SM{major}.x."
            )
        if auto:
            logger.info("attention: sdpa (SM%d.x, no FA support on pre-Ampere)", major)
        return requested or "sdpa"

    if major >= 9:
        # Hopper and newer: FA3 preferred over FA2.
        if auto:
            if _flash_attn3_available():
                logger.info("attention: flash_attention_3 (SM%d.x)", major)
                return "flash_attention_3"
            if _flash_attn_available():
                logger.info("attention: flash_attention_2 (SM%d.x, flash-attn v2)", major)
                return "flash_attention_2"
            logger.info("attention: sdpa (SM%d.x; install flash-attn for FA3/FA2 speedup)", major)
            return "sdpa"
        if requested == "flash_attention_3" and not _flash_attn3_available():
            raise ValueError(
                "flash_attention_3 requires flash-attn >= 3.x. Run: pip install flash-attn"
            )
        if requested == "flash_attention_2" and not _flash_attn_available():
            raise ValueError("flash_attention_2 requires flash-attn. Run: pip install flash-attn")
        return requested  # type: ignore[return-value]

    # Ampere / Ada (SM 8.x): FA2 only, no FA3 hardware support.
    if auto:
        if _flash_attn_available():
            logger.info("attention: flash_attention_2 (SM%d.x)", major)
            return "flash_attention_2"
        logger.info("attention: sdpa (SM%d.x; install flash-attn for FA2 speedup)", major)
        return "sdpa"
    if requested == "flash_attention_3":
        raise ValueError(
            f"flash_attention_3 requires CUDA compute capability >= 9.0 (Hopper). "
            f"Device {device_name!r} is SM{major}.x. Use flash_attention_2 instead."
        )
    if requested == "flash_attention_2" and not _flash_attn_available():
        raise ValueError("flash_attention_2 requires flash-attn. Run: pip install flash-attn")
    return requested  # type: ignore[return-value]


class BGEM3Encoder:
    """Three-stage BGE-M3 encoder backed by transformers AutoModel.

    Stages are designed to run in separate threads:
        encode_pre  — tokenise on CPU (thread-safe, each thread owns its tokenizer)
        encode_core — GPU forward pass (single-thread only)
        encode_post — convert tensors to Python lists (CPU, thread-safe)
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        use_fp16: bool = True,
        torch_compile: bool = False,
        max_length: int = 8192,
        attn_implementation: str | None = None,
    ) -> None:
        """Load model weights and resolve the attention implementation.

        *attn_implementation* may be one of 'flash_attention_2', 'flash_attention_3',
        'sdpa', or 'eager'. Pass None (default) to auto-select the fastest option
        supported by the current hardware and installed packages.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Tokenizer -- template is deepcopied per thread inside _get_tokenizer.
        self._tokenizer_template = AutoTokenizer.from_pretrained(model_name)
        self._local = threading.local()
        self._max_length = max_length

        dtype = torch.float16 if (use_fp16 and device != "cpu") else torch.float32
        resolved_attn = _resolve_attn_implementation(device, dtype, attn_implementation)

        # Load directly in the target precision to avoid a full fp32 allocation + cast.
        load_dtype = dtype if dtype != torch.float32 else None
        _backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            attn_implementation=resolved_attn,
        )
        hidden_size = int(getattr(_backbone.config, "hidden_size"))
        self._backbone: nn.Module = _backbone

        # Sparse linear head -- Linear(hidden_size, 1), weights shipped with the model.
        self._sparse_linear = nn.Linear(hidden_size, 1)
        sparse_pt = (
            os.path.join(model_name, "sparse_linear.pt")
            if os.path.isdir(model_name)
            else hf_hub_download(repo_id=model_name, filename="sparse_linear.pt")
        )
        self._sparse_linear.load_state_dict(
            torch.load(sparse_pt, map_location="cpu", weights_only=True)
        )

        # Backbone dtype is already set via torch_dtype above; only sparse_linear needs casting.
        if use_fp16 and device != "cpu":
            self._sparse_linear.half()
        self._backbone.to(device)
        self._sparse_linear.to(device)
        self._backbone.eval()
        self._sparse_linear.eval()

        if torch_compile and device.startswith("cuda"):
            self._backbone = torch.compile(self._backbone, dynamic=True)  # type: ignore[assignment]

    def _get_tokenizer(self) -> Any:
        """Return a tokenizer private to the calling thread, creating it on first use."""
        if not hasattr(self._local, "tok"):
            self._local.tok = copy.deepcopy(self._tokenizer_template)
        return self._local.tok

    def encode_pre(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenise *texts* on CPU. Safe to call from multiple threads."""
        tok = self._get_tokenizer()
        return cast(
            dict[str, torch.Tensor],
            tok(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_length,
            ),
        )

    @torch.inference_mode()
    def encode_core(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the GPU forward pass. Must be called from a single thread."""
        features = {k: v.to(self.device) for k, v in features.items()}
        last_hidden_state = self._backbone(**features, return_dict=True).last_hidden_state
        dense = F.normalize(last_hidden_state[:, 0], dim=-1)  # CLS pooling + L2 normalise
        sparse = torch.relu(self._sparse_linear(last_hidden_state))  # [B, L, 1]
        return {
            "dense": dense.detach().cpu(),
            "sparse": sparse.squeeze(-1).detach().cpu(),  # [B, L]
            "input_ids": features["input_ids"].cpu(),
        }

    def encode_post(self, raw: dict[str, torch.Tensor], return_sparse: bool) -> EmbeddingResult:
        """Convert raw tensors to an EmbeddingResult. Safe to call from multiple threads."""
        dense = raw["dense"].float().tolist()

        if not return_sparse:
            return EmbeddingResult(dense=dense, sparse_indices=None, sparse_weights=None)

        sparse_indices, sparse_weights = [], []
        for weights_row, ids_row in zip(raw["sparse"], raw["input_ids"]):
            mask = weights_row > 0
            sparse_indices.append(ids_row[mask].tolist())
            sparse_weights.append(weights_row[mask].float().tolist())

        return EmbeddingResult(
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_weights=sparse_weights,
        )

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Return the token count for each text (excluding special tokens)."""
        encoded = self._get_tokenizer()(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]
