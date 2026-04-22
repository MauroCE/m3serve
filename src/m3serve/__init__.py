"""High-throughput BGE-M3 inference engine with dense + sparse embeddings"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import EmbeddingResult

if TYPE_CHECKING:
    from .batcher import Engine

__all__ = ["Engine", "EmbeddingResult"]


def __getattr__(name: str) -> object:
    if name == "Engine":
        from .batcher import Engine

        return Engine
    raise AttributeError(f"module 'm3serve' has no attribute {name!r}")
