import pytest
import torch

from m3serve.model import BGEM3Encoder
from m3serve.types import EmbeddingResult


class BareEncoder(BGEM3Encoder):
    """BGEM3Encoder with __init__ skipped so encode_post can be tested without loading a model."""

    def __init__(self) -> None:
        pass


class TestEncodePost:
    """Tests for BGEM3Encoder.encode_post — no model loading required."""

    @pytest.fixture()
    def encoder(self) -> BareEncoder:
        return BareEncoder()

    def _raw(self, batch: int = 2, seq_len: int = 5, dense_dim: int = 4) -> dict[str, torch.Tensor]:
        sparse = torch.zeros(batch, seq_len)
        sparse[0, 1] = 0.8
        sparse[0, 3] = 0.3
        sparse[1, 2] = 0.5
        return {
            "dense": torch.rand(batch, dense_dim),
            "sparse": sparse,
            "input_ids": torch.arange(batch * seq_len).reshape(batch, seq_len),
        }

    def test_dense_only_returns_no_sparse(self, encoder: BareEncoder) -> None:
        raw = self._raw()
        result = encoder.encode_post(raw, return_sparse=False)
        assert result.sparse_indices is None
        assert result.sparse_weights is None

    def test_dense_shape(self, encoder: BareEncoder) -> None:
        raw = self._raw(batch=3, dense_dim=4)
        result = encoder.encode_post(raw, return_sparse=False)
        assert len(result.dense) == 3
        assert all(len(row) == 4 for row in result.dense)

    def test_sparse_filters_zero_weights(self, encoder: BareEncoder) -> None:
        raw = self._raw()
        result = encoder.encode_post(raw, return_sparse=True)
        assert result.sparse_indices is not None
        assert result.sparse_weights is not None
        # Only positions with weight > 0 should appear
        assert len(result.sparse_indices[0]) == 2
        assert len(result.sparse_indices[1]) == 1

    def test_sparse_indices_match_weights(self, encoder: BareEncoder) -> None:
        raw = self._raw()
        result = encoder.encode_post(raw, return_sparse=True)
        assert result.sparse_indices is not None
        assert result.sparse_weights is not None
        for indices, weights in zip(result.sparse_indices, result.sparse_weights):
            assert len(indices) == len(weights)

    def test_sparse_weights_are_positive(self, encoder: BareEncoder) -> None:
        raw = self._raw()
        result = encoder.encode_post(raw, return_sparse=True)
        assert result.sparse_weights is not None
        for weights in result.sparse_weights:
            assert all(w > 0 for w in weights)

    def test_result_is_embedding_result(self, encoder: BareEncoder) -> None:
        raw = self._raw()
        result = encoder.encode_post(raw, return_sparse=False)
        assert isinstance(result, EmbeddingResult)
