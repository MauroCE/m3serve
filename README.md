# m3serve
[![PyPI](https://img.shields.io/pypi/v/m3serve)](https://pypi.org/project/m3serve/)
[![Python](https://img.shields.io/pypi/pyversions/m3serve)](https://pypi.org/project/m3serve/)
[![CI](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml/badge.svg)](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight async inference engine for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) that returns **dense and sparse embeddings in a single call** — enabling hybrid retrieval without the overhead of a full LLM framework.

## Install

```bash
pip install m3serve
```

## Usage

```python
from m3serve import Engine

engine = Engine(model_name="BAAI/bge-m3", use_fp16=True)
await engine.start()

result = await engine.embed(["hello world"], return_sparse=True)
# result.dense            -> list[list[float]]  (1024-dim)
# result.sparse_indices   -> list[list[int]]    (token ids with non-zero weight)
# result.sparse_weights   -> list[list[float]]  (corresponding weights)

await engine.stop()
```

## How it works

Three background threads run in a pipeline so the GPU is never idle waiting for tokenisation or post-processing:

```
Thread 1  encode_pre   tokenise on CPU        ──┐
Thread 2  encode_core  GPU forward pass    ◄──┘  └──►
Thread 3  encode_post  convert to Python lists       └──► resolved Future
```

Incoming requests are queued and batched by token length (shorter sequences first) to minimise padding waste. Each `embed()` call is a coroutine that returns as soon as its batch is processed — no polling, no callbacks.

## Options

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"BAAI/bge-m3"` | Any bge-m3 compatible model |
| `device` | auto-detected | `"cuda:0"`, `"mps"`, `"cpu"` |
| `use_fp16` | `True` | Half-precision inference (ignored on CPU) |
| `torch_compile` | `False` | `torch.compile` the backbone (CUDA only, adds warmup) |
| `max_batch_size` | `256` | Maximum sequences per GPU batch |
| `batch_delay` | `0.005` | Coalescing window in seconds — sleep after first item arrives to let concurrent requests accumulate. Set to ~½ × GPU inference time for your batch size. |
| `tokenizer_threads` | `4` | Number of threads dedicated to tokenization (`token_lengths`). Each thread holds its own tokenizer copy; all are pre-warmed at `start()` so no cold deepcopy happens during serving. |
| `max_length` | `8192` | Maximum token length per sequence. Longer inputs are truncated. Lower values reduce memory usage and improve throughput for short-text workloads. |

## Tuning `batch_delay`

When the queue goes from empty to non-empty, the preprocess thread sleeps for
`batch_delay` seconds before consuming it. Any requests that arrive during
that window get merged into the same GPU batch.

- **Low concurrency / latency-sensitive:** use `Engine(batch_delay=0)`.
  At c=1 the window is wasted because there is no one else to wait for.
- **High concurrency / throughput-focused:** keep the default (`0.005`).
  Concurrent requests coalesce into larger batches, amortising the GPU's
  fixed per-forward-pass cost.

A good starting value is roughly half your typical GPU inference time.
This heuristic is also used by
[Infinity-emb](https://github.com/michaelfeil/infinity) and mirrors
Triton's [`max_queue_delay_microseconds`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

## Limitations

**Single model, single GPU.** m3serve runs one bge-m3 instance on one GPU.
There is no replica support or multi-GPU sharding.

**Coalescing window adds latency at low concurrency.** The default
`batch_delay=0.005` sleeps 5 ms after the first request arrives to let
concurrent requests accumulate into a larger batch. At c=1 this sleep is
always wasted, adding ~5 ms to every request. Use `Engine(batch_delay=0)`
for single-client or latency-sensitive workloads.

**p99 latency can be spiky at medium concurrency.** A request that just
misses a coalescing window must wait for the next cycle. In practice this
means p99 can be 5-10x higher than p50 at moderate concurrency levels
(e.g. c=4 to c=8). If your workload has strict p99 SLAs, benchmark under
your expected traffic pattern before deploying.

**bge-m3 only.** The engine is purpose-built for `BAAI/bge-m3` and models
with the same three-stage encode interface. It is not a general-purpose
inference server.
