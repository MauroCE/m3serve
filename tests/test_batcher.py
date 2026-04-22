import asyncio
from collections.abc import AsyncGenerator

import pytest
import torch

from m3serve.batcher import Engine
from m3serve.model import BGEM3Encoder
from m3serve.types import EmbeddingResult


class FakeEncoder(BGEM3Encoder):
    """BGEM3Encoder replacement returning deterministic tensors without loading a model."""

    DENSE_DIM = 8

    def __init__(self) -> None:
        pass

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]

    def encode_pre(self, texts: list[str]) -> dict[str, torch.Tensor]:
        n = len(texts)
        seq_len = max(len(t.split()) for t in texts)
        return {
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }

    def encode_core(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n, seq_len = features["input_ids"].shape
        sparse = torch.zeros(n, seq_len)
        sparse[:, 0] = 0.9  # one non-zero weight per row
        return {
            "dense": torch.ones(n, self.DENSE_DIM),
            "sparse": sparse,
            "input_ids": features["input_ids"],
        }

    def encode_post(self, raw: dict[str, torch.Tensor], return_sparse: bool) -> EmbeddingResult:
        dense = raw["dense"].float().tolist()
        if not return_sparse:
            return EmbeddingResult(dense=dense)
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


@pytest.fixture()
async def engine() -> AsyncGenerator[Engine, None]:
    e = Engine(_encoder=FakeEncoder())
    await e.start()
    yield e
    await e.stop()


class TestEngine:
    async def test_embed_returns_embedding_result(self, engine: Engine) -> None:
        result = await engine.embed(["hello world"])
        assert isinstance(result, EmbeddingResult)

    async def test_dense_shape(self, engine: Engine) -> None:
        result = await engine.embed(["one two three", "four five"])
        assert len(result.dense) == 2
        assert all(len(row) == FakeEncoder.DENSE_DIM for row in result.dense)

    async def test_sparse_none_by_default(self, engine: Engine) -> None:
        result = await engine.embed(["hello world"])
        assert result.sparse_indices is None
        assert result.sparse_weights is None

    async def test_sparse_returned_when_requested(self, engine: Engine) -> None:
        result = await engine.embed(["hello world"], return_sparse=True)
        assert result.sparse_indices is not None
        assert result.sparse_weights is not None
        assert len(result.sparse_indices) == 1

    async def test_concurrent_requests_resolved(self, engine: Engine) -> None:
        results = await asyncio.gather(
            engine.embed(["first"]),
            engine.embed(["second"]),
            engine.embed(["third"]),
        )
        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)

    async def test_batch_with_multiple_texts(self, engine: Engine) -> None:
        texts = ["one", "two three", "four five six", "seven"]
        result = await engine.embed(texts)
        assert len(result.dense) == len(texts)
