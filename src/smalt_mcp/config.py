# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Static configuration. Pure leaf module — no internal imports.

Env-driven, mirroring the deco-assaying pattern. One `Config` dataclass with
nested `EmbeddingConfig` so the embedder's `make_embedder(cfg)` contract
(`cfg.embedding.provider`, `cfg.embedding.model`, `cfg.embedding.dim`) works
unchanged from the ported code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    VERSION: str = version("smalt-mcp")
except PackageNotFoundError:  # editable install before first build
    VERSION = "0.0.0+local"

# ---- HTTP server ----

PORT: int = int(os.environ.get("PORT", "35833"))  # 35833 = deco-assaying's 35832 + 1
HOST: str = os.environ.get("HOST", "0.0.0.0")


# E-2 (concurrency): default thread-pool worker count. Tool handlers
# dispatch blocking work via `asyncio.to_thread`, which uses the loop's
# default `ThreadPoolExecutor`. Python's default is `min(32, cpu_count + 4)`
# — 32 is the empirical ceiling for typical I/O-bound workloads. Operators
# can tune via `SMALT_THREAD_POOL_WORKERS` for heavier deployments.
_DEFAULT_THREAD_POOL_WORKERS = 32


# ---- structured config (for the embedder etc.) ----


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dim: int


@dataclass(frozen=True)
class Config:
    """Runtime config bundle. Construct once at startup; pass to everything that needs it."""

    smalt_dir: Path
    embedding: EmbeddingConfig
    # E-2 (concurrency): max worker threads in the asyncio loop's
    # default `ThreadPoolExecutor`. Bounds the number of concurrent
    # handler executions (each `asyncio.to_thread` call takes one
    # worker for the duration of the blocking work). Default 32.
    thread_pool_workers: int = _DEFAULT_THREAD_POOL_WORKERS


def load_config() -> Config:
    """Build a Config from environment variables."""
    smalt_dir = Path(os.environ.get("SMALT_DIR", "~/Documents/Smalt")).expanduser().resolve()
    embedding = EmbeddingConfig(
        provider=os.environ.get("EMBEDDING_PROVIDER", "fastembed"),
        model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        dim=int(os.environ.get("EMBEDDING_DIM", "384")),
    )
    thread_pool_workers = int(os.environ.get("SMALT_THREAD_POOL_WORKERS", str(_DEFAULT_THREAD_POOL_WORKERS)))
    if thread_pool_workers <= 0:
        thread_pool_workers = _DEFAULT_THREAD_POOL_WORKERS
    return Config(
        smalt_dir=smalt_dir,
        embedding=embedding,
        thread_pool_workers=thread_pool_workers,
    )
