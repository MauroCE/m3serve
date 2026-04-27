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

    def encode_core(
        self,
        features: dict[str, torch.Tensor],
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, torch.Tensor]:
        n, seq_len = features["input_ids"].shape
        raw: dict[str, torch.Tensor] = {"dense": torch.ones(n, self.DENSE_DIM)}
        if return_sparse:
            sparse = torch.zeros(n, seq_len)
            sparse[:, 0] = 0.9  # one non-zero weight per row
            raw["sparse"] = sparse
            raw["input_ids"] = features["input_ids"]
        if return_colbert:
            raw["colbert"] = torch.rand(n, max(seq_len - 1, 1), self.DENSE_DIM)
            raw["attention_mask"] = features["attention_mask"]
        return raw

    def encode_post(
        self,
        raw: dict[str, torch.Tensor],
        return_sparse: bool,
        return_colbert: bool = False,
    ) -> EmbeddingResult:
        dense = raw["dense"].float().tolist()
        sparse_indices = None
        sparse_weights = None
        colbert_vecs = None
        if return_sparse:
            sparse_indices, sparse_weights = [], []
            for weights_row, ids_row in zip(raw["sparse"], raw["input_ids"]):
                mask = weights_row > 0
                sparse_indices.append(ids_row[mask].tolist())
                sparse_weights.append(weights_row[mask].float().tolist())
        if return_colbert:
            colbert_vecs = []
            for vecs, mask_row in zip(raw["colbert"], raw["attention_mask"][:, 1:]):
                n_real = int(mask_row.sum().item())
                colbert_vecs.append(vecs[:n_real].float().tolist())
        return EmbeddingResult(
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_weights=sparse_weights,
            colbert_vecs=colbert_vecs,
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

    async def test_colbert_none_by_default(self, engine: Engine) -> None:
        result = await engine.embed(["hello world"])
        assert result.colbert_vecs is None

    async def test_colbert_returned_when_requested(self, engine: Engine) -> None:
        result = await engine.embed(["hello world"], return_colbert=True)
        assert result.colbert_vecs is not None
        assert len(result.colbert_vecs) == 1
        assert all(len(vec) == FakeEncoder.DENSE_DIM for vec in result.colbert_vecs[0])

    async def test_thread_death_raises_not_hangs(self) -> None:
        class BrokenEncoder(FakeEncoder):
            def encode_core(
                self,
                features: dict[str, torch.Tensor],
                return_sparse: bool = True,
                return_colbert: bool = False,
            ) -> dict[str, torch.Tensor]:
                raise RuntimeError("simulated GPU crash")

        e = Engine(_encoder=BrokenEncoder())
        await e.start()
        with pytest.raises(RuntimeError, match="simulated GPU crash"):
            await e.embed(["hello"])
        await e.stop()
