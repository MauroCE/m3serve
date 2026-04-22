import asyncio
import threading
import time

from m3serve.queue import LengthSortedQueue
from m3serve.types import _QueueItem


def make_item(max_length: int, texts: list[str] | None = None) -> _QueueItem:
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    loop.close()
    return _QueueItem(
        texts=texts or ["text"],
        return_sparse=False,
        future=future,
        max_length=max_length,
    )


class TestLengthSortedQueue:
    def test_empty_length(self) -> None:
        assert len(LengthSortedQueue()) == 0

    def test_put_increments_length(self) -> None:
        q = LengthSortedQueue()
        q.put(make_item(1))
        assert len(q) == 1

    def test_pop_batch_returns_all_items_when_under_limit(self) -> None:
        q = LengthSortedQueue()
        q.put(make_item(1))
        q.put(make_item(2))
        batch = q.pop_batch(max_size=10)
        assert len(batch) == 2
        assert len(q) == 0

    def test_pop_batch_respects_max_size(self) -> None:
        q = LengthSortedQueue()
        for i in range(5):
            q.put(make_item(i))
        batch = q.pop_batch(max_size=3)
        assert len(batch) == 3
        assert len(q) == 2

    def test_pop_batch_sorted_shortest_first(self) -> None:
        q = LengthSortedQueue()
        for length in [10, 2, 7, 1, 5]:
            q.put(make_item(length))
        batch = q.pop_batch(max_size=10)
        lengths = [item.max_length for item in batch]
        assert lengths == sorted(lengths)

    def test_pop_batch_returns_empty_on_timeout(self) -> None:
        q = LengthSortedQueue()
        batch = q.pop_batch(max_size=10, timeout=0.02)
        assert batch == []

    def test_pop_batch_wakes_when_item_arrives(self) -> None:
        q = LengthSortedQueue()
        results: list[list[_QueueItem]] = []

        def consumer() -> None:
            results.append(q.pop_batch(max_size=10, timeout=2.0))

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)
        q.put(make_item(1))
        t.join(timeout=1.0)

        assert len(results) == 1
        assert len(results[0]) == 1
