"""The indexer: markdown corpus → LanceDB.

Walks `smalt/pages/`, parses every markdown file's YAML frontmatter, validates
it against the `Page` Pydantic union, and projects pages + links + claims +
embeddings into LanceDB. Refreshes the FTS and ANN indexes at the end of a run.

**Discipline.** This module accepts every shared resource as an argument
(`smalt_root`, `embedder`, `db`). Nothing is stored as a module-level global.
The server constructs a single `Embedder` at startup and reuses it across
indexer calls.

**Embedding length.** Long bodies are truncated to a fixed character budget
before being embedded. `BAAI/bge-small-en-v1.5` accepts up to 512 tokens
(~2k chars) — beyond that fastembed truncates internally anyway. Real
chunking is planned for a later iteration.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from smalt_mcp.schema import Page
from smalt_mcp.storage import lance, paths
from smalt_mcp.storage.markdown import ParsedPage, iter_page_files, parse_page

if TYPE_CHECKING:
    import lancedb

    from smalt_mcp.storage.embedder import Embedder

logger = logging.getLogger(__name__)


# Body characters fed to the embedder. Anything past this is truncated.
#
# This is a *conservative guess* — bge-small-en-v1.5 counts subword tokens, not
# characters, with a 512-token model context. A naive English-text estimate of
# ~4 chars/token would suggest a ~2k-char budget; we use 2000 to leave headroom
# for tokenization variance. fastembed truncates internally past the model's
# real context, so passing more than this is safe-but-wasteful, not unsafe.
EMBED_BODY_CHAR_BUDGET = 2000


@dataclass
class IndexResult:
    """Summary of one `Indexer.run()` call."""

    scanned: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0  # unchanged since last run
    deleted: int = 0  # rows in the index whose backing files are gone
    failed: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
    cancelled: bool = False

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable summary for MCP responses."""
        return {
            "scanned": self.scanned,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "deleted": self.deleted,
            "failed": self.failed,
            "failures": [(str(p), msg) for p, msg in self.failures],
            "duration_seconds": self.duration_seconds,
            "cancelled": self.cancelled,
        }


class Indexer:
    """Project the markdown corpus into LanceDB.

    Construction is explicit so callers decide how to obtain the embedder and
    DB connection. The Indexer itself never touches `Config` directly — it
    only sees the resources handed to it.
    """

    def __init__(
        self,
        *,
        smalt_root: Path,
        embedder: Embedder,
        db: lancedb.DBConnection,
        progress: Callable[[str], None] | None = None,
        progress_event: Callable[[dict[str, object]], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        progress : optional callable; receives human-readable progress lines
            (one per scanned file) for terminal-style logging.
        progress_event : optional callable; receives structured per-phase
            updates as dicts (`phase`, counters, `current_file`). The caller
            can wire this to a task-progress channel so MCP clients see live
            progress.
        is_cancelled : optional callable; when it returns True, the indexer
            stops at its next checkpoint (between files). The indexer never
            depends on any specific scheduler — the caller adapts its cancel
            signal to this callable.
        """
        self.smalt_root = smalt_root.expanduser().resolve()
        self.embedder = embedder
        self.db = db
        self.progress = progress or (lambda _: None)
        self.progress_event = progress_event or (lambda _: None)
        self.is_cancelled = is_cancelled or (lambda: False)
        self.model_version = embedder.model_version

    # ---- public API ----

    def run(self, *, full: bool = False) -> IndexResult:
        """Walk pages, project changed ones into LanceDB, refresh indexes."""
        started = perf_counter()
        result = IndexResult()

        files = iter_page_files(paths.pages_dir(self.smalt_root))
        result.scanned = len(files)
        self._emit_progress(result, phase="scanning", current_file=None)

        existing_hashes: dict[str, str] = {} if full else lance.existing_page_hashes(self.db)
        seen_ids: set[str] = set()

        # Parse each file; collect those that need (re)indexing.
        to_index: list[ParsedPage] = []
        for path in files:
            if self.is_cancelled():
                result.cancelled = True
                self.progress("  [cancel] cancellation requested; stopping")
                self._emit_progress(result, phase="cancelled", current_file=None)
                break

            try:
                parsed = parse_page(path, smalt_root=self.smalt_root)
            except (ValidationError, ValueError, OSError) as e:
                result.failed += 1
                result.failures.append((path, _short_error(e)))
                self.progress(f"  [fail] {path.name}: {_short_error(e)}")
                self._emit_progress(result, phase="scanning", current_file=str(path.name))
                continue

            page_id = parsed.frontmatter.id
            seen_ids.add(page_id)

            old_hash = existing_hashes.get(page_id)
            if old_hash == parsed.content_hash:
                result.skipped += 1
                self._emit_progress(result, phase="scanning", current_file=parsed.rel_path)
                continue
            if old_hash is None:
                result.inserted += 1
                self.progress(f"  [new]  {parsed.rel_path}")
            else:
                result.updated += 1
                self.progress(f"  [chg]  {parsed.rel_path}")
            self._emit_progress(result, phase="scanning", current_file=parsed.rel_path)
            to_index.append(parsed)

        # Pages indexed previously whose files are gone.
        for stale_id in set(existing_hashes) - seen_ids:
            self._delete_page(stale_id)
            result.deleted += 1
            self.progress(f"  [del]  {stale_id}")
            self._emit_progress(result, phase="scanning", current_file=stale_id)

        # Embed bodies in one batch and project to LanceDB.
        if to_index:
            self._emit_progress(result, phase="indexing", current_file=None)
            self._project(to_index)
            self._emit_progress(result, phase="refreshing", current_file=None)
            self._refresh_indexes()

        result.duration_seconds = perf_counter() - started
        self._emit_progress(result, phase="done", current_file=None)
        return result

    # ---- progress helper ----

    def _emit_progress(
        self,
        result: IndexResult,
        *,
        phase: str,
        current_file: str | None,
    ) -> None:
        """Publish a structured progress event for any subscribed listener."""
        self.progress_event(
            {
                "phase": phase,
                "scanned": result.scanned,
                "inserted": result.inserted,
                "updated": result.updated,
                "skipped": result.skipped,
                "deleted": result.deleted,
                "failed": result.failed,
                "current_file": current_file,
            }
        )

    # ---- internals ----

    def _project(self, parsed_pages: Iterable[ParsedPage]) -> None:
        """Write rows for `parsed_pages` into pages / embeddings / links / claims."""
        items = list(parsed_pages)
        if not items:
            return

        now = datetime.now(UTC)
        existing_created_at = self._fetch_created_at({p.frontmatter.id for p in items})

        # Embeddings — one batch call.
        bodies = [_truncate_for_embedding(p.body) for p in items]
        vectors = self.embedder.embed(bodies)

        page_rows: list[dict[str, Any]] = []
        embedding_rows: list[dict[str, Any]] = []
        # strict=True so a misbehaving embedder that returns the wrong
        # number of vectors fails loudly here rather than silently truncating
        # the index. Misalignment between pages and vectors would mean the
        # vector for page N actually belongs to page N+k somewhere downstream.
        for p, vec in zip(items, vectors, strict=True):
            page_rows.append(_page_row(p, now=now, existing_created_at=existing_created_at))
            embedding_rows.append(
                {
                    "page_id": p.frontmatter.id,
                    "vector": vec,
                    "model_version": self.model_version,
                }
            )

        lance.upsert_pages(self.db, page_rows)
        lance.upsert_embeddings(self.db, embedding_rows)

        # Per-page replace for owned tables.
        for p in items:
            page_id = p.frontmatter.id
            link_rows = _link_rows(p.frontmatter, page_id=page_id)
            claim_rows = _claim_rows(p.frontmatter, page_id=page_id)
            lance.replace_links_for_page(self.db, page_id, link_rows)
            lance.replace_claims_for_page(self.db, page_id, claim_rows)

    def _delete_page(self, page_id: str) -> None:
        """Remove a page and its owned rows from the index."""
        quoted = lance.sql_str(page_id)
        pages = self.db.open_table(lance.TABLE_PAGES)
        pages.delete(f"id = {quoted}")
        embeddings = self.db.open_table(lance.TABLE_EMBEDDINGS)
        embeddings.delete(f"page_id = {quoted}")
        links = self.db.open_table(lance.TABLE_LINKS)
        links.delete(f"from_id = {quoted}")
        claims = self.db.open_table(lance.TABLE_CLAIMS)
        claims.delete(f"page_id = {quoted}")

    def _refresh_indexes(self) -> None:
        """Rebuild FTS and ANN after a batch of writes."""
        try:
            lance.create_or_refresh_fts(self.db)
        except Exception as e:  # pragma: no cover — best effort
            logger.warning("FTS index refresh failed: %s", e)
        try:
            lance.create_or_refresh_vector_index(self.db)
        except Exception as e:  # pragma: no cover — best effort
            logger.warning("Vector index refresh failed: %s", e)

    def _fetch_created_at(self, page_ids: set[str]) -> dict[str, datetime]:
        """Look up existing `created_at` so updates preserve the original value.

        Delegates to `lance.fetch_created_at` which uses an `id IN (...)`
        filter and a column projection — neither row count nor body size
        affects how much we transfer.
        """
        raw = lance.fetch_created_at(self.db, page_ids)
        out: dict[str, datetime] = {}
        for pid, ts in raw.items():
            # pyarrow gives us a python datetime for timestamp("us", tz="UTC");
            # be defensive and coerce anything missing tzinfo to UTC.
            py_ts: datetime = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if py_ts.tzinfo is None:
                py_ts = py_ts.replace(tzinfo=UTC)
            out[pid] = py_ts
        return out


# ---- row builders ----


def _page_row(
    parsed: ParsedPage, *, now: datetime, existing_created_at: dict[str, datetime]
) -> dict[str, Any]:
    fm = parsed.frontmatter
    created = existing_created_at.get(fm.id, now)
    return {
        "id": fm.id,
        "path": parsed.rel_path,
        "type": fm.type.value,
        "title": fm.title,
        "body": parsed.body,
        "frontmatter_json": json.dumps(parsed.raw_frontmatter, default=str, sort_keys=True),
        "content_hash": parsed.content_hash,
        "created_at": created,
        "updated_at": now,
    }


def _link_rows(page: Page, *, page_id: str) -> list[dict[str, Any]]:
    return [
        {
            "from_id": page_id,
            "to_id": link.target,
            "label": link.label,
            "source_page": link.via_source,
        }
        for link in page.links_out
    ]


def _claim_rows(page: Page, *, page_id: str) -> list[dict[str, Any]]:
    # All four Page subtypes (entity, concept, source, synthesis) declare
    # `claims: list[Claim]` directly; no need for getattr-with-default.
    rows: list[dict[str, Any]] = []
    for c in page.claims:
        # Coerce any typed value to a string for the columnar table; the
        # `value_type` column lets readers decode it.
        value_str = None if c.value is None else str(c.value)
        rows.append(
            {
                "id": c.id,
                "page_id": page_id,
                "claim_text": c.text,
                "value_type": c.value_type,
                "value": value_str,
                "unit": c.unit,
                "confidence": c.confidence,
                "confidence_label": c.confidence_label.value,
                "source_ref": c.source_ref,
            }
        )
    return rows


# ---- small utilities ----


def _truncate_for_embedding(body: str) -> str:
    if len(body) <= EMBED_BODY_CHAR_BUDGET:
        return body
    return body[:EMBED_BODY_CHAR_BUDGET]


def _short_error(e: BaseException) -> str:
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    if len(text) > 200:
        text = text[:200] + "…"
    return text
