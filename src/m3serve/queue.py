"""Thread-safe queue that yields batches sorted by token length (shortest first)."""

import threading

from .types import _PrioritizedItem, _QueueItem


class LengthSortedQueue:
    """Thread-safe queue that yields batches sorted by token length (shortest first).

    Shorter sequences are batched together to minimise padding waste.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[_PrioritizedItem] = []
        self._event = threading.Event()

    def __len__(self) -> int:
        """Return the number of pending items."""
        with self._lock:
            return len(self._items)

    def put(self, item: _QueueItem) -> None:
        """Enqueue *item* and wake any thread blocked in pop_batch."""
        with self._lock:
            self._items.append(_PrioritizedItem(priority=item.max_length, item=item))
            self._event.set()

    def pop_batch(self, max_size: int, timeout: float = 0.05) -> list[_QueueItem]:
        """Wait up to *timeout* seconds, then return up to *max_size* items sorted by length."""
        if not self._items:
            self._event.wait(timeout)
        with self._lock:
            if not self._items:
                return []
            self._items.sort()
            batch, self._items = self._items[:max_size], self._items[max_size:]
            if not self._items:
                self._event.clear()
        return [p.item for p in batch]
