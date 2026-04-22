"""Async BGE-M3 inference engine with dynamic batching and pipelined GPU execution."""

import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from .model import BGEM3Encoder
from .queue import LengthSortedQueue
from .types import EmbeddingResult, _QueueItem

# (features_or_raw, items, return_sparse)
_Payload = tuple[dict[str, torch.Tensor], list[_QueueItem], bool]

_TIMEOUT = 0.05
logger = logging.getLogger(__name__)


class Engine:
    """Async BGE-M3 inference engine with dynamic batching and pipelined GPU execution.

    Three background threads run concurrently:
        _preprocess_thread  — tokenises incoming requests (CPU)
        _core_thread        — runs the GPU forward pass (single thread)
        _postprocess_thread — converts tensors to Python objects and resolves futures (CPU)

    Exceptions inside any stage are caught per-batch: affected futures are
    rejected immediately and the thread continues serving subsequent requests.

    Usage::

        engine = Engine(model_name="BAAI/bge-m3", use_fp16=True)
        await engine.start()
        result = await engine.embed(["hello world"], return_sparse=True)
        await engine.stop()
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        use_fp16: bool = True,
        torch_compile: bool = False,
        max_batch_size: int = 256,
        _encoder: BGEM3Encoder | None = None,
    ) -> None:
        self._encoder = _encoder or BGEM3Encoder(model_name, device, use_fp16, torch_compile)
        self._max_batch_size = max_batch_size
        self._request_queue: LengthSortedQueue = LengthSortedQueue()
        self._feature_queue: queue.Queue[_Payload] = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue[_Payload] = queue.Queue(maxsize=4)
        self._shutdown = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="m3serve")
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Start background inference threads. Must be called before embed()."""
        self._loop = asyncio.get_running_loop()
        self._shutdown.clear()
        for fn in (self._preprocess_thread, self._core_thread, self._postprocess_thread):
            self._pool.submit(fn)

    async def stop(self) -> None:
        """Signal threads to stop and wait for them to finish."""
        self._shutdown.set()
        await asyncio.to_thread(self._pool.shutdown, wait=True)

    async def embed(
        self,
        texts: list[str],
        return_sparse: bool = False,
        timeout: float = 60.0,
    ) -> EmbeddingResult:
        """Embed *texts*, optionally returning sparse (lexical) weights alongside dense vectors.

        Raises asyncio.TimeoutError if no result is returned within *timeout* seconds,
        which guards against silent thread death.
        """
        assert self._loop is not None, "Call start() before embed()."
        lengths = await self._loop.run_in_executor(None, self._encoder.token_lengths, texts)
        future: asyncio.Future[EmbeddingResult] = self._loop.create_future()
        self._request_queue.put(
            _QueueItem(
                texts=texts,
                return_sparse=return_sparse,
                future=future,
                max_length=max(lengths, default=0),
            )
        )
        return await asyncio.wait_for(future, timeout=timeout)

    # -- background threads --------------------------------------------------

    def _preprocess_thread(self) -> None:
        while not self._shutdown.is_set():
            batch = self._request_queue.pop_batch(self._max_batch_size, _TIMEOUT)
            if not batch:
                continue
            try:
                groups: dict[bool, list[_QueueItem]] = {}
                for item in batch:
                    groups.setdefault(item.return_sparse, []).append(item)
                for return_sparse, items in groups.items():
                    all_texts = [t for item in items for t in item.texts]
                    features = self._encoder.encode_pre(all_texts)
                    self._enqueue(self._feature_queue, (features, items, return_sparse))
            except Exception as exc:
                logger.exception("encode_pre failed")
                self._reject(batch, exc)

    def _core_thread(self) -> None:
        while not self._shutdown.is_set():
            try:
                features, items, return_sparse = self._feature_queue.get(timeout=_TIMEOUT)
            except queue.Empty:
                continue
            try:
                raw = self._encoder.encode_core(features)
                self._enqueue(self._result_queue, (raw, items, return_sparse))
            except Exception as exc:
                logger.exception("encode_core failed")
                self._reject(items, exc)

    def _postprocess_thread(self) -> None:
        while not self._shutdown.is_set():
            try:
                raw, items, return_sparse = self._result_queue.get(timeout=_TIMEOUT)
            except queue.Empty:
                continue
            try:
                offset = 0
                for item in items:
                    n = len(item.texts)
                    sliced = {k: v[offset : offset + n] for k, v in raw.items()}
                    result = self._encoder.encode_post(sliced, item.return_sparse)
                    offset += n
                    assert self._loop is not None
                    self._loop.call_soon_threadsafe(item.future.set_result, result)
            except Exception as exc:
                logger.exception("encode_post failed")
                self._reject(items, exc)

    def _reject(self, items: list[_QueueItem], exc: Exception) -> None:
        """Resolve *items* futures with *exc* so callers get an exception, not a hang."""
        assert self._loop is not None
        for item in items:
            if not item.future.done():
                self._loop.call_soon_threadsafe(item.future.set_exception, exc)

    def _enqueue(self, target: queue.Queue[_Payload], payload: _Payload) -> None:
        """Put *payload* into *target*, retrying on full until shutdown."""
        while not self._shutdown.is_set():
            try:
                target.put(payload, timeout=_TIMEOUT)
                return
            except queue.Full:
                continue
