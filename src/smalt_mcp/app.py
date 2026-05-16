"""Shared-resource bundle for the running smalt-mcp server.

One `App` instance is constructed at startup (in `server.lifespan`) and made
available to every tool handler. It owns:

- `cfg`: the loaded `Config` (Smalt path, embedding settings)
- `db`: a single LanceDB connection (lazily, only if the Smalt dir exists)
- `embedder`: a single fastembed model (lazy — first use loads it)
- `mutex`: the single-writer corpus-write mutex

The lazy connection / embedder construction lets the server start even when
`SMALT_DIR` doesn't exist yet — `status` can still report "not initialized"
without crashing. Real bootstrap of an empty Smalt dir happens via the
`bootstrap` tool (planned).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smalt_mcp.config import Config, load_config
from smalt_mcp.mutex import CorpusWriteMutex

if TYPE_CHECKING:
    import lancedb

    from smalt_mcp.storage.embedder import Embedder


class App:
    """Shared resources for the running server."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg: Config = cfg if cfg is not None else load_config()
        self.mutex: CorpusWriteMutex = CorpusWriteMutex()
        self._db: lancedb.DBConnection | None = None
        self._embedder: Embedder | None = None

    # ---- lazy resource accessors ----

    def db(self) -> lancedb.DBConnection:
        """Return the LanceDB connection, opening it on first use.

        Raises `FileNotFoundError` if the configured Smalt dir doesn't exist
        yet — callers that need it should bootstrap first.
        """
        if self._db is None:
            from smalt_mcp.storage.lance import connect

            if not self.cfg.smalt_dir.exists():
                raise FileNotFoundError(
                    f"Smalt directory does not exist: {self.cfg.smalt_dir}. "
                    f"Bootstrap it first (smalt.bootstrap)."
                )
            self._db = connect(self.cfg.smalt_dir)
        return self._db

    def embedder(self) -> Embedder:
        """Return the fastembed model, constructing it on first use (slow first call)."""
        if self._embedder is None:
            from smalt_mcp.storage.embedder import make_embedder

            self._embedder = make_embedder(self.cfg)
        return self._embedder

    # ---- status helpers ----

    def smalt_exists(self) -> bool:
        return self.cfg.smalt_dir.exists()
