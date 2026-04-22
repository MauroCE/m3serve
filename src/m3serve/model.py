"""BGE-M3 encoder: tokenise → GPU forward pass → post-process."""

import copy
import os
import threading
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from .types import EmbeddingResult


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
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Tokenizer — template is deepcopied per thread inside _get_tokenizer.
        self._tokenizer_template = AutoTokenizer.from_pretrained(model_name)
        self._local = threading.local()

        # Backbone: XLM-RoBERTa transformer.
        _backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(getattr(_backbone.config, "hidden_size"))
        self._backbone: nn.Module = _backbone

        # Sparse linear head — Linear(hidden_size, 1), weights shipped with the model.
        self._sparse_linear = nn.Linear(hidden_size, 1)
        sparse_pt = (
            os.path.join(model_name, "sparse_linear.pt")
            if os.path.isdir(model_name)
            else hf_hub_download(repo_id=model_name, filename="sparse_linear.pt")
        )
        self._sparse_linear.load_state_dict(
            torch.load(sparse_pt, map_location="cpu", weights_only=True)
        )

        if use_fp16 and device != "cpu":
            self._backbone.half()
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
                max_length=tok.model_max_length,
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
