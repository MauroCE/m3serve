"""Thin wrapper around BGEM3FlagModel exposing a three-stage encode interface."""

import copy
from typing import cast

import torch
from FlagEmbedding import BGEM3FlagModel

from .types import EmbeddingResult


class BGEM3Encoder:
    """Thin wrapper around BGEM3FlagModel exposing a three-stage encode interface.

    Stages are designed to run in separate threads:
        encode_pre  — tokenise on CPU (thread-safe, uses a private tokenizer copy)
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

        self._embedder = BGEM3FlagModel(
            model_name_or_path=model_name,
            use_fp16=use_fp16 and device != "cpu",
            devices=device,
        )
        # Private tokenizer copy avoids state corruption across concurrent threads.
        self._tokenizer = copy.deepcopy(self._embedder.tokenizer)
        self._model = self._embedder.model  # EncoderOnlyEmbedderM3ModelForInference
        if use_fp16 and device != "cpu":
            self._model.half()
        self._model.to(device)
        self._model.eval()

        if torch_compile and device.startswith("cuda"):
            self._model.model = torch.compile(self._model.model, dynamic=True)

    def encode_pre(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenise *texts* on CPU. Safe to call from multiple threads."""
        return cast(
            dict[str, torch.Tensor],
            self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._tokenizer.model_max_length,
            ),
        )

    @torch.inference_mode()
    def encode_core(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the GPU forward pass. Must be called from a single thread."""
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self._model(
            text_input=features,
            return_dense=True,
            return_sparse=True,
            return_sparse_embedding=False,  # returns per-token weights [B, L, 1]
        )
        return {
            "dense": output["dense_vecs"].detach().cpu(),
            "sparse": output["sparse_vecs"].squeeze(-1).detach().cpu(),  # [B, L]
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
        encoded = self._tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]
