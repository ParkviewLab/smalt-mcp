"""Embedding-model abstraction.

Only the local `fastembed` provider is implemented; hosted providers (`voyage`,
`openai`) are sketched as placeholders for later milestones.

The `Embedder` protocol is what the rest of the codebase depends on, so
swapping providers is a single-point change. The Indexer takes an `Embedder`
instance — it doesn't construct one — so the server can build a single
Embedder at startup and reuse it across all index calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from smalt_mcp.config import Config


class Embedder(Protocol):
    """The contract everything in smalt-mcp talks to for embeddings."""

    @property
    def dim(self) -> int: ...

    @property
    def model_version(self) -> str:
        """A stable identifier of the model — `<provider>:<model>` — stored alongside vectors so we know what produced them."""
        ...

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one float-list per input."""
        ...


class FastembedEmbedder:
    """Local ONNX-backed embedder via the fastembed library."""

    def __init__(self, model_name: str, *, dim: int) -> None:
        # Lazy import so test environments without fastembed installed
        # don't pay the import cost just to import this module.
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._dim = dim
        self._impl = TextEmbedding(model_name)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_version(self) -> str:
        return f"fastembed:{self._model_name}"

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        # fastembed accepts any iterable and yields numpy arrays one at a
        # time. We materialize the input only once (when fastembed iterates
        # it internally), and convert each output array to a plain list so
        # downstream consumers (LanceDB, JSON, tests) don't pull in numpy.
        return [np.asarray(v, dtype=np.float32).tolist() for v in self._impl.embed(texts)]


def make_embedder(cfg: Config) -> Embedder:
    """Construct the embedder configured by `cfg.embedding`. Single source
    of truth for which provider gets used."""
    provider = cfg.embedding.provider.lower()
    if provider == "fastembed":
        return FastembedEmbedder(cfg.embedding.model, dim=cfg.embedding.dim)
    if provider in ("voyage", "openai"):
        raise NotImplementedError(
            f"embedding provider {provider!r} is supported in config schema "
            f"but not yet wired up — only `fastembed` is implemented."
        )
    raise ValueError(f"unknown embedding provider: {provider!r}")
