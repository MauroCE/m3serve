# m3serve
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml/badge.svg)](https://github.com/MauroCE/m3serve/actions/workflows/ci.yml)

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
