# m3serve
[![PyPI](https://img.shields.io/pypi/v/m3serve)](https://pypi.org/project/m3serve/)
[![Python](https://img.shields.io/pypi/pyversions/m3serve)](https://pypi.org/project/m3serve/)
[![CI](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml/badge.svg)](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Lightweight async inference engine for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) that returns **dense, sparse, and ColBERT multi-vector embeddings in a single call**.

## Install

```bash
pip install m3serve
```

## Usage

```python
from m3serve import Engine

engine = Engine(model_name="BAAI/bge-m3", use_fp16=True)
await engine.start()

result = await engine.embed(["hello world"], return_sparse=True, return_colbert=True)
# result.dense            -> list[list[float]]        (1024-dim, one vector per text)
# result.sparse_indices   -> list[list[int]]          (token ids with non-zero weight)
# result.sparse_weights   -> list[list[float]]        (corresponding weights)
# result.colbert_vecs     -> list[list[list[float]]]  (one 1024-dim vector per token, padding stripped)

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
| `attn_implementation` | `None` (auto) | Attention backend: `"flash_attention_2"`, `"flash_attention_3"`, `"sdpa"`, or `"eager"`. `None` auto-selects the fastest supported option. |
| `return_sparse` | `False` | Return sparse (lexical) weights alongside dense vectors. Passed to `embed()`, not `Engine()`. |
| `return_colbert` | `False` | Return ColBERT multi-vector embeddings alongside dense vectors. Passed to `embed()`, not `Engine()`. |

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

## Flash Attention

[Flash Attention](https://github.com/Dao-AILab/flash-attention) is an optimised attention algorithm that is significantly faster and more memory-efficient than standard attention, especially for long sequences. m3serve supports [FA2](https://arxiv.org/abs/2307.08691) (Ampere GPUs and newer, e.g. A100, RTX 3090) and [FA3](https://arxiv.org/abs/2407.08608) (Hopper GPUs, e.g. H100).

By default, m3serve **auto-selects the best available option** for your hardware, no configuration needed. If `flash-attn` is installed and your GPU supports it, FA2 or FA3 will be used automatically. Otherwise it falls back to PyTorch's built-in [SDPA](https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) kernel, which is always available.

To enable FA2/FA3, install the optional dependency:

```bash
pip install m3serve[flash-attn]
```

You can also control the attention implementation explicitly:

```python
# Auto (default) picks the best option for your hardware
engine = Engine(attn_implementation=None)

# Force a specific backend
engine = Engine(attn_implementation="flash_attention_2")
engine = Engine(attn_implementation="flash_attention_3")
engine = Engine(attn_implementation="sdpa")
engine = Engine(attn_implementation="eager")
```

If you request FA2/FA3 but your GPU or packages cannot support it, m3serve raises a clear error explaining exactly what is missing. The hardware check uses `torch.cuda.get_device_capability()` to inspect the GPU's compute capability directly — no guesswork.

| Backend | Hardware required | Package required |
|---|---|---|
| `flash_attention_3` | Hopper+ (SM 9.0, e.g. H100) | `flash-attn >= 3.x` |
| `flash_attention_2` | Ampere+ (SM 8.0, e.g. A100, RTX 3090) | `flash-attn` |
| `sdpa` | Any (CPU, MPS, CUDA) | none |
| `eager` | Any | none |

Note: FA2/FA3 require `transformers>=5` (where XLM-Roberta got refactored onto `AttentionInterface`)
 on `transformers<5` the engine silently falls back to SDPA.
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
-  `m3serve` GPU memory profiling across batch sizes, sequence lengths, and concurrency. [Colab Notebook](https://colab.research.google.com/drive/1qNUjQSk3Nj65sSFS94LEUJd9MU72HuyY?usp=sharing)

# Benchmark Results
Benchmark results on Colab's free T4 GPU. Better hardware should lead to better results.

- **Up to 58% higher throughput than FlagEmbedding** at every batch size (1705 vs 1079 t/s at batch=128), thanks to the pipelined CPU/GPU architecture.
- **Dynamic batching absorbs concurrency with minimal latency cost**: p50 rises from 18.9ms at c=1 to just 31.7ms at c=32, while throughput scales ~19x.
- **p99 latency stays under 64ms up to c=32**, but the 5ms coalescing window adds fixed overhead at low concurrency; use `Engine(batch_delay=0)` for single-client workloads.

```bash
===========================================================================
BATCH SIZE SWEEP  (baseline, sequential)
===========================================================================
baseline  batch_size=1                    throughput=   22.6 t/s  p50=  32.7ms  p99= 230.5ms
baseline  batch_size=8                    throughput=  188.7 t/s  p50=  36.6ms  p99=  68.2ms
baseline  batch_size=32                   throughput=  801.7 t/s  p50=  40.1ms  p99=  49.3ms
baseline  batch_size=64                   throughput=  930.2 t/s  p50=  72.3ms  p99=  74.3ms
baseline  batch_size=128                  throughput= 1083.6 t/s  p50= 132.2ms  p99= 134.3ms

===========================================================================
BATCH SIZE SWEEP  (m3serve, concurrency=1)
===========================================================================
m3serve   batch_size=1                    throughput=   42.6 t/s  p50=  19.1ms  p99=  43.2ms
m3serve   batch_size=8                    throughput=  381.1 t/s  p50=  19.8ms  p99=  34.1ms
m3serve   batch_size=32                   throughput= 1292.1 t/s  p50=  25.2ms  p99=  28.6ms
m3serve   batch_size=64                   throughput= 1603.9 t/s  p50=  41.7ms  p99=  43.6ms
m3serve   batch_size=128                  throughput= 1893.4 t/s  p50=  75.6ms  p99=  79.0ms

===========================================================================
CONCURRENCY SWEEP  (m3serve, batch_size=1 per caller)
===========================================================================
m3serve   concurrency=1                   throughput=   46.6 t/s  p50=  19.2ms  p99=  33.9ms
m3serve   concurrency=2                   throughput=   90.6 t/s  p50=  19.7ms  p99=  33.0ms
m3serve   concurrency=4                   throughput=  179.9 t/s  p50=  20.7ms  p99=  37.1ms
m3serve   concurrency=8                   throughput=  354.4 t/s  p50=  22.2ms  p99=  32.4ms
m3serve   concurrency=16                  throughput=  629.9 t/s  p50=  25.2ms  p99=  34.4ms
m3serve   concurrency=32                  throughput=  946.5 t/s  p50=  34.0ms  p99=  44.8ms
```
