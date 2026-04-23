# m3serve
[![PyPI](https://img.shields.io/pypi/v/m3serve)](https://pypi.org/project/m3serve/)
[![Python](https://img.shields.io/pypi/pyversions/m3serve)](https://pypi.org/project/m3serve/)
[![CI](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml/badge.svg)](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Lightweight async inference engine for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) that returns **dense and sparse embeddings in a single call**.

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

Incoming requests are queued and batched by token length (shorter sequences first) to minimise padding waste. Each `embed()` call is a coroutine that returns as soon as its batch is processed: no polling, no callbacks.

## Options

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"BAAI/bge-m3"` | Any bge-m3 compatible model |
| `device` | auto-detected | `"cuda:0"`, `"mps"`, `"cpu"` |
| `use_fp16` | `True` | Half-precision inference (ignored on CPU) |
| `torch_compile` | `False` | `torch.compile` the backbone (CUDA only, adds warmup) |
| `max_batch_size` | `256` | Maximum sequences per GPU batch |
| `batch_delay` | `0.005` | Coalescing window in seconds — sleep after first item arrives to let concurrent requests accumulate. Set to ~0.5 x GPU inference time for your batch size. |
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

# Notebooks / Tutorials
- `m3serve` generates identical sparse and dense vectors to FlagEmbedding. [Colab Notebook](https://colab.research.google.com/drive/1StwAXVNOLWYPkH0Gng_1pVxFkDE5hcy_?usp=sharing)
- `m3serve` benchmark against FlagEmbedding on Colab's free T4 GPU. [Colab Notebook](https://colab.research.google.com/drive/12ROy5q6YoGnNzEsY8WMi-86HJXCnXyjF?usp=sharing)

# Benchmark Results
Benchmark results on Colab's free T4 GPU. Better hardware should lead to better results.

- **Up to 58% higher throughput than FlagEmbedding** at every batch size (1705 vs 1079 t/s at batch=128), thanks to the pipelined CPU/GPU architecture.
- **Dynamic batching absorbs concurrency with minimal latency cost**: p50 rises from 18.9ms at c=1 to just 31.7ms at c=32, while throughput scales ~19x.
- **p99 latency stays under 64ms up to c=32**, but the 5ms coalescing window adds fixed overhead at low concurrency; use `Engine(batch_delay=0)` for single-client workloads.

```bash
===========================================================================
BATCH SIZE SWEEP  (baseline, sequential)
===========================================================================
baseline  batch_size=1                    throughput=   27.7 t/s  p50=  33.0ms  p99=  59.1ms
baseline  batch_size=8                    throughput=  216.3 t/s  p50=  36.1ms  p99=  51.5ms
baseline  batch_size=32                   throughput=  807.5 t/s  p50=  39.6ms  p99=  52.1ms
baseline  batch_size=64                   throughput=  932.3 t/s  p50=  71.3ms  p99=  74.2ms
baseline  batch_size=128                  throughput= 1079.0 t/s  p50= 132.7ms  p99= 135.0ms

===========================================================================
BATCH SIZE SWEEP  (m3serve, concurrency=1)
===========================================================================
m3serve   batch_size=1                    throughput=   38.8 t/s  p50=  20.0ms  p99=  66.4ms
m3serve   batch_size=8                    throughput=  360.2 t/s  p50=  20.5ms  p99=  54.3ms
m3serve   batch_size=32                   throughput= 1035.6 t/s  p50=  30.9ms  p99=  41.4ms
m3serve   batch_size=64                   throughput= 1419.7 t/s  p50=  46.5ms  p99=  54.9ms
m3serve   batch_size=128                  throughput= 1705.0 t/s  p50=  81.9ms  p99=  90.1ms

===========================================================================
CONCURRENCY SWEEP  (m3serve, batch_size=1 per caller)
===========================================================================
m3serve   concurrency=1                   throughput=   47.4 t/s  p50=  18.9ms  p99=  32.4ms
m3serve   concurrency=2                   throughput=   89.5 t/s  p50=  19.5ms  p99=  33.8ms
m3serve   concurrency=4                   throughput=  194.7 t/s  p50=  20.1ms  p99=  28.3ms
m3serve   concurrency=8                   throughput=  356.0 t/s  p50=  21.8ms  p99=  34.6ms
m3serve   concurrency=16                  throughput=  538.8 t/s  p50=  28.6ms  p99=  38.6ms
m3serve   concurrency=32                  throughput=  906.3 t/s  p50=  31.7ms  p99=  63.6ms
```
