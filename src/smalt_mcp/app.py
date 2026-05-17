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

C-8 observability: `App` also tracks runtime state used by the
`index_status` MCP tool + `/admin/health` HTTP route — last indexer run
metadata + per-index build state (FTS / ANN). Updated via
`record_indexer_run()` which `tools._run_indexer` calls after each pass.

C-13 async tasks: `App.scheduler` lazily constructs a `Scheduler` for
long-running async operations (`reindex_all`, future heavy ops).
Started in the FastAPI lifespan; shut down on app exit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from smalt_mcp.config import Config, load_config
from smalt_mcp.mutex import CorpusWriteMutex
from smalt_mcp.scheduler import Scheduler

if TYPE_CHECKING:
    import lancedb

    from smalt_mcp.storage.embedder import Embedder
    from smalt_mcp.storage.indexer import IndexResult


class App:
    """Shared resources for the running server."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg: Config = cfg if cfg is not None else load_config()
        self.mutex: CorpusWriteMutex = CorpusWriteMutex()
        self._db: lancedb.DBConnection | None = None
        self._embedder: Embedder | None = None
        # C-13 async tasks: scheduler is constructed eagerly because
        # it doesn't need a running event loop at __init__ time —
        # only `enqueue` / `start_gc` do.
        self.scheduler: Scheduler = Scheduler()

        # C-8: indexer-run + index-build observability state. None until
        # the first indexer pass completes (via _run_indexer in tools.py).
        self.last_indexer_run_at: datetime | None = None
        self.last_indexer_run_duration_seconds: float | None = None
        # `last_indexer_result` is the full IndexResult.to_dict() of the
        # most recent run — useful for /admin/health to surface per-run
        # counts (inserted/updated/deleted/failed) without re-deriving.
        self.last_indexer_result: dict[str, Any] | None = None
        # Per-index build state from the most recent refresh that
        # touched them. Shapes match what lance.create_or_refresh_fts /
        # create_or_refresh_vector_index return.
        self.last_fts_status: dict[str, dict[str, Any]] | None = None
        self.last_vector_status: dict[str, Any] | None = None

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

    # ---- C-8 observability ----

    def record_indexer_run(self, result: IndexResult) -> None:
        """Update observability state after an indexer pass completes.

        Called by `tools._run_indexer` after every pass. Only updates the
        FTS / vector status when the corresponding refresh actually ran
        — a no-op indexer run (zero pages changed) leaves the previous
        index state intact, which is the truthful state for
        `index_status` to report.
        """
        self.last_indexer_run_at = datetime.now(UTC)
        self.last_indexer_run_duration_seconds = result.duration_seconds
        self.last_indexer_result = result.to_dict()
        if result.fts_status is not None:
            self.last_fts_status = result.fts_status
        if result.vector_status is not None:
            self.last_vector_status = result.vector_status

    def index_status_payload(self) -> dict[str, Any]:
        """Assemble the full `index_status` / `/admin/health` payload.

        Walks LanceDB (cheaply — count_rows per table) + reads the
        observability state set by `record_indexer_run`. Safe to call
        even before any indexer pass has run (returns nulls for the
        not-yet-known fields).

        Payload shape (stable surface — both the MCP tool and the HTTP
        route return this):

        ```
        {
          "smalt_dir": str,
          "smalt_exists": bool,
          "tables": {<table_name>: {"row_count": int}},
          "indexer": {
            "last_run_at": ISO datetime | None,
            "last_run_duration_seconds": float | None,
            "last_result": IndexResult.to_dict() | None,
          },
          "indexes": {
            "fts": {<field>: {"status": "ok"|"failed", "error": str | None}} | None,
            "vector": {"status": "ok"|"failed"|"skipped", "error": str | None, "reason": str | None} | None,
          },
          "embedding": {provider, model, dim, model_loaded: bool},
          "mutex": {locked, holder, acquire_count, total_wait_seconds, mean_wait_ms},
        }
        ```
        """
        from smalt_mcp.storage import lance

        smalt_dir = str(self.cfg.smalt_dir)
        smalt_exists = self.smalt_exists()

        tables: dict[str, dict[str, Any]] = {}
        if smalt_exists:
            try:
                db = self.db()
                existing = set(lance.list_tables(self.cfg.smalt_dir))
                for name in lance.ALL_TABLES:
                    if name not in existing:
                        continue
                    try:
                        tables[name] = {"row_count": db.open_table(name).count_rows()}
                    except Exception as e:  # noqa: BLE001 — surface, don't crash status
                        tables[name] = {"row_count": None, "error": f"{type(e).__name__}: {e}"}
            except FileNotFoundError:
                # Smalt dir exists but index/lance/ doesn't — pre-bootstrap.
                pass

        return {
            "smalt_dir": smalt_dir,
            "smalt_exists": smalt_exists,
            "tables": tables,
            "indexer": {
                "last_run_at": self.last_indexer_run_at.isoformat() if self.last_indexer_run_at else None,
                "last_run_duration_seconds": self.last_indexer_run_duration_seconds,
                "last_result": self.last_indexer_result,
            },
            "indexes": {
                "fts": self.last_fts_status,
                "vector": self.last_vector_status,
            },
            "embedding": {
                "provider": self.cfg.embedding.provider,
                "model": self.cfg.embedding.model,
                "dim": self.cfg.embedding.dim,
                "model_loaded": self._embedder is not None,
            },
            "mutex": {
                "locked": self.mutex.locked,
                "holder": self.mutex.holder,
                "acquire_count": self.mutex.acquire_count,
                "total_wait_seconds": round(self.mutex.total_wait_seconds, 6),
                "mean_wait_ms": round(self.mutex.mean_wait_ms, 3),
            },
        }
