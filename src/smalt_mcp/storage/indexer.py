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
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import frontmatter
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


# ---- canonical IndexPages ----
#
# Two auto-generated IndexPages are regenerated at the end of every indexer
# run from the current `pages` table contents. Listed here as `(filename,
# id, title, flag)`. New canonical IndexPages can be added in later C PRs
# (entities, sources, etc.) by appending to this list. User-defined custom
# IndexPages (planned for a future PR) will load their stored_query from
# disk; this constant is just the bootstrap-time defaults.
_CANONICAL_INDEX_PAGES: tuple[tuple[str, str, str, str], ...] = (
    ("glossary.md", "idx-glossary", "Glossary", "glossary"),
    ("domains.md",  "idx-domains",  "Domains",  "is_domain"),
)


# Glossary body length cap per entry — keeps the generated file readable
# even with hundreds of glossary terms. The first sentence of the
# concept's body is included; longer entries are elided.
_INDEX_ENTRY_DEFINITION_MAX_CHARS = 300


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

        # Regenerate the canonical IndexPages from the current pages
        # table (after the run's writes have landed). Each IndexPage that
        # actually changed gets re-projected so callers see the updated
        # body immediately via list_pages / read_page.
        self._emit_progress(result, phase="regenerating_indexes", current_file=None)
        index_changed = self._regenerate_index_pages()
        if index_changed:
            self._project(index_changed)
            # IndexPages are concepts in the FTS+ANN indexes too — refresh
            # so a search() right after a write_page that updated an index
            # returns the new body.
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

    def _regenerate_index_pages(self) -> list[ParsedPage]:
        """Rewrite the canonical IndexPages from current corpus state.

        Returns the list of IndexPages whose on-disk content actually
        changed (or were created); these must be projected to LanceDB so
        list_pages / read_page surface the new bodies. Skipped IndexPages
        (already up-to-date) are not returned.
        """
        pages_dir = paths.pages_dir(self.smalt_root)
        pages_dir.mkdir(parents=True, exist_ok=True)

        # Pull all concept pages in one query; partition by flag client-side.
        concept_entries = self._collect_concept_entries()

        changed: list[ParsedPage] = []
        for filename, page_id, title, flag in _CANONICAL_INDEX_PAGES:
            entries = concept_entries.get(flag, [])
            target = pages_dir / filename
            new_content = _format_index_page(
                page_id=page_id,
                title=title,
                flag=flag,
                entries=entries,
            )
            if _file_content_matches(target, new_content):
                continue
            _atomic_write_text(target, new_content)
            try:
                parsed = parse_page(target, smalt_root=self.smalt_root)
            except (ValidationError, ValueError, OSError) as e:
                # Shouldn't happen: we just wrote a well-formed file. Log
                # and skip projection; next run will retry.
                logger.warning("regenerated IndexPage %s failed to parse: %s", filename, e)
                continue
            changed.append(parsed)
        return changed

    def _collect_concept_entries(self) -> dict[str, list[dict[str, Any]]]:
        """Walk the pages table for `type='concept'` rows; bucket by flag.

        Returns `{flag: [{id, title, body, domains, ...}, ...]}` for each
        flag declared in `_CANONICAL_INDEX_PAGES` (only `glossary` and
        `is_domain` today). The buckets are sorted by `id` so the
        regenerated body is deterministic across runs (and across
        machines — important for back-merge cascade hygiene).
        """
        try:
            pages_table = self.db.open_table(lance.TABLE_PAGES)
        except FileNotFoundError:
            return {}

        arrow = (
            pages_table.search()
            .where(f"type = {lance.sql_str('concept')}")
            .select(["id", "title", "body", "frontmatter_json"])
            .limit(100_000)  # honest cap; if a Smalt has >100k concept pages, IndexPages
                              # need a different rendering strategy than "every entry inline"
            .to_arrow()
        )

        bucket: dict[str, list[dict[str, Any]]] = {flag: [] for _, _, _, flag in _CANONICAL_INDEX_PAGES}

        for i in range(arrow.num_rows):
            pid = arrow.column("id")[i].as_py()
            title = arrow.column("title")[i].as_py()
            body = arrow.column("body")[i].as_py() or ""
            fm_raw = arrow.column("frontmatter_json")[i].as_py()
            try:
                fm = json.loads(fm_raw) if fm_raw else {}
            except json.JSONDecodeError:
                continue
            entry = {
                "id": pid,
                "title": title,
                "body": body,
                "domains": list(fm.get("domains") or []),
            }
            for flag in bucket:
                if fm.get(flag) is True:
                    bucket[flag].append(entry)

        for flag in bucket:
            bucket[flag].sort(key=lambda e: e["id"])
        return bucket

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
    # Entity / Concept / Source / Synthesis declare `claims: list[Claim]`
    # directly. IndexPage doesn't carry claims (auto-generated pages have
    # no per-page assertions to track); handle that case by returning [].
    claims = getattr(page, "claims", None) or []
    rows: list[dict[str, Any]] = []
    for c in claims:
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


# ---- IndexPage regeneration helpers ----


def _format_index_page(
    *,
    page_id: str,
    title: str,
    flag: str,
    entries: list[dict[str, Any]],
) -> str:
    """Build the full text content (frontmatter + body) for an IndexPage.

    Body format:
      - Sorted markdown list, one bullet per entry.
      - Each bullet: `- **<title>** (<id>[, <domain1>, <domain2>]) — <definition>`
        where the trailing definition is the first sentence of the
        concept's body (truncated at `_INDEX_ENTRY_DEFINITION_MAX_CHARS`).
      - Empty corpus → header + a "no entries yet" note.

    The output is a frontmatter+markdown string ready for atomic write.
    """
    fm: dict[str, Any] = {
        "id": page_id,
        "type": "index",
        "title": title,
        "auto_generated": True,
        "stored_query": {"kind": "concept_flag", "flag": flag},
    }

    if not entries:
        body = (
            f"# {title}\n\n"
            f"_Auto-generated by the indexer. No entries yet — "
            f"add a concept page with `{flag}: true` and re-run an indexer "
            f"pass (any write_page call auto-triggers one)._\n"
        )
    else:
        lines: list[str] = [f"# {title}", "", f"_Auto-generated by the indexer. {len(entries)} entries._", ""]
        for entry in entries:
            ident = entry["id"]
            entry_title = entry["title"]
            domains = entry["domains"]
            paren_parts = [ident, *domains]
            paren = ", ".join(paren_parts)
            bullet = f"- **{entry_title}** ({paren})"
            definition = _first_sentence(entry["body"], _INDEX_ENTRY_DEFINITION_MAX_CHARS)
            if definition:
                bullet += f" — {definition}"
            lines.append(bullet)
        lines.append("")
        body = "\n".join(lines)

    post = frontmatter.Post(body)
    post.metadata.update(fm)
    return frontmatter.dumps(post)


def _first_sentence(body: str, max_chars: int) -> str:
    """Pull the first sentence (up to max_chars) from a page body.

    Strips leading whitespace; treats the first `. `, `.\\n`, or end of body
    as the sentence break. Returns "" if the body is empty.
    """
    text = body.lstrip()
    if not text:
        return ""
    # Find the first sentence-ending period followed by whitespace or end.
    # Simple heuristic — good enough for glossary-entry intros.
    end = len(text)
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 == len(text) or text[i + 1] in " \t\n"):
            end = i + 1
            break
    first = text[:end].strip()
    if len(first) > max_chars:
        first = first[: max_chars - 1].rstrip() + "…"
    return first


def _file_content_matches(path: Path, expected: str) -> bool:
    """True if `path` exists and its UTF-8 contents equal `expected`."""
    if not path.exists():
        return False
    try:
        return path.read_text(encoding="utf-8") == expected
    except (OSError, UnicodeDecodeError):
        return False


def _atomic_write_text(target: Path, content: str) -> None:
    """Tmp-then-rename write; same shape as the tools.py write helper.

    Lives here (duplicated) to keep `indexer.py` free of `tools.py`
    imports (circular). Both helpers should evolve together if the write
    invariants change.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)
