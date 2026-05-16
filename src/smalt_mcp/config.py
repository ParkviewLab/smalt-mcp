"""Static configuration. Pure leaf module — no internal imports.

Env-driven, mirroring the deco-assaying pattern. One `Config` dataclass with
nested `EmbeddingConfig` so the embedder's `make_embedder(cfg)` contract
(`cfg.embedding.provider`, `cfg.embedding.model`, `cfg.embedding.dim`) works
unchanged from the ported code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    VERSION: str = version("smalt-mcp")
except PackageNotFoundError:  # editable install before first build
    VERSION = "0.0.0+local"

# ---- HTTP server ----

PORT: int = int(os.environ.get("PORT", "35833"))  # 35833 = deco-assaying's 35832 + 1
HOST: str = os.environ.get("HOST", "0.0.0.0")


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
    embedding: EmbeddingConfig = field(default=None)  # type: ignore[assignment]


def load_config() -> Config:
    """Build a Config from environment variables."""
    smalt_dir = Path(os.environ.get("SMALT_DIR", "~/Documents/Smalt")).expanduser().resolve()
    embedding = EmbeddingConfig(
        provider=os.environ.get("EMBEDDING_PROVIDER", "fastembed"),
        model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        dim=int(os.environ.get("EMBEDDING_DIM", "384")),
    )
    return Config(smalt_dir=smalt_dir, embedding=embedding)
