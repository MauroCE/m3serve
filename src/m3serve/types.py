"""Types shared between batcher.py and queue.py."""

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Output of a single embed() call."""

    dense: list[list[float]]
    sparse_indices: list[list[int]] | None = None
    sparse_weights: list[list[float]] | None = None
    colbert_vecs: list[list[list[float]]] | None = None


@dataclass
class _QueueItem:
    """A single embed() request waiting to be processed."""

    texts: list[str]
    return_sparse: bool
    future: asyncio.Future  # type: ignore[type-arg]
    max_length: int = 0  # longest token count in texts
    return_colbert: bool = False


@dataclass(order=True)
class _PrioritizedItem:
    """Wraps a _QueueItem with a sort key for length-sorted batching."""

    priority: int  # token length, shorter sequences first
    item: _QueueItem = field(compare=False)
