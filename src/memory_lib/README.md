# memory_lib

A clean, standalone memory layer library for an AI agent proxy.

## Overview

`memory_lib` provides a cognitive memory system with:

- **Observer pipeline** — processes incoming text through filtering, NER, embedding, and scoring
- **Matrix search** — cosine-similarity search over embedding vectors
- **SQLite persistence** — durable RAM↔disk sync with batched writes
- **Memory consolidation** — background decay, score recalculation, eviction
- **Experience learning** — long-term behavioral pattern tracking per topic
- **Anchor system** — permanent skeletal records of significant events

## Quick Start

```bash
# Install
pip install -e .

# Run the server
python -m memory_lib.main
# or
uv memory_lib.api.routes:app --host 127.0.0.1 --port 8766
```

## Project Structure

```
memory_lib/
  api/            — HTTP API endpoints (FastAPI)
  config.py       — Configuration (Pydantic settings)
  domain/         — Core domain types and Result pattern
  main.py         — Server entry point
  memory/         — Session index, search index, scoring, dissolver, consolidation
  models/         — ONNX model management, embedder, NER, reranker
  observer/       — Observation pipeline: filter, marker, NER, entities
  storage/        — SQLite schemas, persistence layer
  subconscious/   — Anchor index and anchor model
```

## License

MIT
