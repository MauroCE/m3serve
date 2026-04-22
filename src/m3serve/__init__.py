"""High-throughput BGE-M3 inference engine with dense + sparse embeddings"""

from .batcher import Engine
from .types import EmbeddingResult

__all__ = ["Engine", "EmbeddingResult"]
