# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tool specs + dispatch.

Each tool is registered as a `ToolDef` with its MCP spec (name, description,
input schema) and a `Scope` (READ_ONLY or READ_WRITE). The server's
`@mcp.list_tools()` filters by the caller's scope; `@mcp.call_tool()`
delegates here via `dispatch()`.

Keeping tool specs + handlers in one module (rather than mixed with the
FastAPI/MCP plumbing) makes it easy to add a new tool: add a `ToolDef`
entry to `TOOLS` plus a handler function. No edits to `server.py` needed.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import frontmatter
from mcp import types
from pydantic import TypeAdapter, ValidationError

from smalt_mcp.permissions import SCOPE_TIER, Scope
from smalt_mcp.scheduler import TERMINAL_STATES, Task, TaskState
from smalt_mcp.schema import Claim, Page, PageType
from smalt_mcp.storage import lance, paths
from smalt_mcp.storage.markdown import parse_page

if TYPE_CHECKING:
    from smalt_mcp.app import App


logger = logging.getLogger(__name__)


PAGE_ADAPTER: TypeAdapter[Page] = TypeAdapter(Page)


# Map a page's type to the subdirectory under `smalt/pages/` where its file lives.
_TYPE_TO_SUBDIR: dict[PageType, str] = {
    PageType.ENTITY: "entities",
    PageType.CONCEPT: "concepts",
    PageType.SOURCE: "sources",
    PageType.SYNTHESIS: "syntheses",
}


# Bootstrap placeholders. These are intentionally minimal — they exist so a
# fresh Smalt has *something* at the canonical paths; downstream agents and
# humans flesh them out over time, treating them as living documents (see
# `cobalt-grinding/docs/north_star.md` → "The Smalt documents itself").
_SCHEMA_MD_PLACEHOLDER = """# SCHEMA.md

This is the human-readable narrative of the page types, frontmatter shape,
and link-edge vocabulary in this Smalt. The machine-readable version lives
in `smalt_mcp/schema.py` (the Pydantic models).

This document is a **living artifact** — schema changes are proposed,
tested, and applied through the `ebony-enriching` MCP server (the lab
notebook substrate); accepted changes land here.
"""

_POLICY_MD_PLACEHOLDER = """# POLICY.md

This is the human-readable policy that agentic systems operate under when
producing or modifying pages in this Smalt: when to create new pages vs.
extend, how contradictions are handled, how confidence is assigned.

Like SCHEMA.md, this document is **living** — policy changes are proposed
and reviewed through the `ebony-enriching` lab-notebook substrate. The
falsifiability / cost-tier discipline for proposals themselves lives in
ebony-enriching's own POLICY.md.
"""


# ---- ToolDef + handler signature ----


Handler = Callable[["App", dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDef:
    """One MCP tool: spec + handler + permission scope."""

    spec: types.Tool
    scope: Scope
    handler: Handler


# ---- E-2 (concurrency): sync-handler → thread-pool wrapper ----
#
# Tool handlers used to be `async def` but call sync code inside
# (LanceDB queries, file I/O, fastembed inference). That blocks the
# event loop for the duration of every operation, so two concurrent
# MCP requests serialize. E-2 introduces this decorator: handlers
# whose bodies are sync are written as plain `def`, decorated with
# `@_wrap_sync_in_thread`, and dispatch via `asyncio.to_thread` to
# the loop's default ThreadPoolExecutor (sized by
# `cfg.thread_pool_workers`, default 32).
#
# The handler signature visible to `dispatch` stays the same — the
# decorator produces an async wrapper, so `await tool.handler(...)`
# works unchanged.
#
# Handlers that genuinely use async/await internally (today only
# `reindex_all`, which uses `await asyncio.to_thread` + the
# scheduler) stay as `async def` and are NOT decorated.


def _wrap_sync_in_thread(
    sync_fn: Callable[[App, dict[str, Any]], dict[str, Any]],
) -> Handler:
    """Turn a sync handler `(app, arguments) -> dict` into an async
    handler that runs the work in the asyncio loop's default
    ThreadPoolExecutor via `asyncio.to_thread`.

    Use as a decorator on every tool handler whose body is sync
    (LanceDB queries, file I/O, fastembed inference, etc.) so that
    concurrent MCP requests don't serialize on the event loop.

    The wrapped handler is reported by `functools.wraps`-preserved
    metadata as the original sync function, so introspection
    (logging, tool-name reporting) is unchanged.
    """

    @functools.wraps(sync_fn)
    async def wrapper(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(sync_fn, app, arguments)

    return wrapper


# ---- shared helpers ----


def _not_initialized() -> dict[str, Any]:
    return {
        "error": "smalt_not_initialized",
        "message": "Smalt directory or LanceDB tables not present; bootstrap first.",
    }


def _ensure_initialized(app: App) -> dict[str, Any] | None:
    """Return an error payload if the Smalt isn't initialized, else `None`.

    Use as: `err = _ensure_initialized(app); if err is not None: return err`.
    """
    if not app.smalt_exists():
        return _not_initialized()
    try:
        app.db()
    except FileNotFoundError:
        return _not_initialized()
    return None


def _serialize_and_write_page(target: Path, fm_dict: dict[str, Any], body: str) -> None:
    """Serialize `{frontmatter: ..., body: ...}` to a YAML+markdown file at `target`, atomically.

    Atomic = write to a sibling `.tmp` file, then `os.replace()` onto the target.
    Caller is responsible for holding any required write-lock (mutex).
    """
    post = frontmatter.Post(body)
    post.metadata.update(fm_dict)
    serialized = frontmatter.dumps(post)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(serialized, encoding="utf-8")
    os.replace(tmp, target)


def _page_target_path(smalt_root: Path, page: Page) -> Path:
    """Compute the canonical on-disk path for `page` inside `smalt_root`.

    Two id shapes (see `schema._validate_id`):

    - **Slug id** (e.g., `ent-alice`, `con-cs`, `src-doc1`): path is
      `pages/<subdir>/<id>.md`. The v0 shape.

    - **Section id** (e.g., `src-foo::src/utils.py`): the M3 hybrid
      layout. `::` translates to `/` to produce a nested file path:
      `pages/sources/src-foo/src/utils.py.md`. The source file's
      original extension (`.py` here) is preserved in the path; the
      `.md` wrapper extension is appended. Use `src-foo::index` for
      the multi-file source's index page → `pages/sources/src-foo/index.md`.
    """
    subdir = _TYPE_TO_SUBDIR[page.type]
    if "::" in page.id:
        rel = page.id.replace("::", "/")
        return paths.pages_dir(smalt_root) / subdir / (rel + ".md")
    return paths.pages_dir(smalt_root) / subdir / f"{page.id}.md"


def _run_indexer(app: App) -> dict[str, Any]:
    """Run an incremental indexer pass. Caller must hold the corpus mutex.

    Also updates the App's observability state (C-8) — `last_indexer_run_at`,
    `last_indexer_result`, `last_fts_status`, `last_vector_status` — so the
    `index_status` tool + `/admin/health` HTTP route can surface what
    happened on the most recent pass.
    """
    from smalt_mcp.storage.indexer import Indexer

    result = Indexer(
        smalt_root=app.cfg.smalt_dir,
        embedder=app.embedder(),
        db=app.db(),
    ).run()
    app.record_indexer_run(result)
    return result.to_dict()


# ---- property-filter helpers (shared by list_pages + search) ----
#
# v0 list_pages and search filtered by `type` and `prefix` only. C-2 adds
# six new property filters that look inside each page's frontmatter (the
# raw on-disk dict, preserved in LanceDB's `pages.frontmatter_json` text
# column):
#
#   glossary: bool                   ← ConceptPage.glossary
#   is_domain: bool                  ← ConceptPage.is_domain
#   domain: str                      ← ConceptPage/SourcePage/EntityPage.domains list-contains
#   fetched_at_before: ISO datetime  ← SourcePage.fetched_at < threshold
#   fetched_at_after: ISO datetime   ← SourcePage.fetched_at > threshold
#   has_aliases_containing: str      ← PageBase.aliases list-contains
#
# All AND-composed. SQL-NULL semantics: a filter on a field the page
# doesn't have → page doesn't match (matches how a SQL `WHERE field = X`
# excludes NULL rows).
#
# Implementation does client-side JSON parsing on `frontmatter_json` —
# same O(N) pattern as `_find_pages_by_alias`. C-10 (aliases LanceDB
# column) sets up the perf foundation; some of these filters may migrate
# to LanceDB-native predicates in later C PRs once the column work lands.

_PROPERTY_FILTER_ARGS: frozenset[str] = frozenset(
    {
        "glossary",
        "is_domain",
        "domain",
        "fetched_at_before",
        "fetched_at_after",
        "has_aliases_containing",
    }
)


def _has_property_filters(args: dict[str, Any]) -> bool:
    """True if the caller supplied any of the property-filter args (not None)."""
    return any(args.get(k) is not None for k in _PROPERTY_FILTER_ARGS)


def _parse_property_filters(
    args: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Parse + validate property-filter args.

    Returns `(filters, None)` on success, or `(None, error_payload)` on
    validation failure. The returned `filters` dict only contains keys for
    filters the caller actually supplied — absent keys mean "no filter on
    that field," not "filter on null."
    """
    filters: dict[str, Any] = {}

    for key in ("glossary", "is_domain"):
        val = args.get(key)
        if val is None:
            continue
        if not isinstance(val, bool):
            return None, {
                "error": "validation_error",
                "message": f"{key} must be a bool; got {type(val).__name__}",
            }
        filters[key] = val

    for key in ("domain", "has_aliases_containing"):
        val = args.get(key)
        if val is None:
            continue
        if not isinstance(val, str) or not val:
            return None, {
                "error": "validation_error",
                "message": f"{key} must be a non-empty string; got {val!r}",
            }
        filters[key] = val

    for key in ("fetched_at_before", "fetched_at_after"):
        val = args.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            return None, {
                "error": "validation_error",
                "message": f"{key} must be an ISO 8601 datetime string; got {type(val).__name__}",
            }
        try:
            filters[key] = datetime.fromisoformat(val)
        except ValueError as e:
            return None, {
                "error": "validation_error",
                "message": f"{key} must be ISO 8601; got {val!r} ({e})",
            }

    return filters, None


def _apply_property_filters(fm: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if `fm` (a page's raw frontmatter dict) passes every filter.

    All filters AND-composed. SQL-NULL semantics: a filter on a field the
    page doesn't have → page doesn't match.

    Bool filters (`glossary`, `is_domain`) compare the raw value as it
    appears in YAML — a ConceptPage that omits `glossary:` will NOT match
    `glossary=False` (the schema default doesn't materialize at filter
    time; the user is filtering on what's literally written). Caveat
    documented in the tool spec.
    """
    if not filters:
        return True

    # Bool fields — exact match on the raw frontmatter value.
    for key in ("glossary", "is_domain"):
        if key in filters and fm.get(key) != filters[key]:
            return False

    # List-contains fields.
    if "domain" in filters:
        domains = fm.get("domains")
        if not isinstance(domains, list) or filters["domain"] not in domains:
            return False
    if "has_aliases_containing" in filters:
        aliases = fm.get("aliases")
        if not isinstance(aliases, list) or filters["has_aliases_containing"] not in aliases:
            return False

    # SourcePage `fetched_at` range. Parse the page's value lazily; missing
    # or unparseable → page doesn't match (matches the SQL-NULL semantics).
    if "fetched_at_before" in filters or "fetched_at_after" in filters:
        fetched_at_raw = fm.get("fetched_at")
        if not fetched_at_raw:
            return False
        try:
            fetched_at = datetime.fromisoformat(str(fetched_at_raw))
        except (ValueError, TypeError):
            return False
        if "fetched_at_before" in filters and fetched_at >= filters["fetched_at_before"]:
            return False
        if "fetched_at_after" in filters and fetched_at <= filters["fetched_at_after"]:
            return False

    return True


# ---- handler: status ----


@_wrap_sync_in_thread
def status(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Report Smalt path, existence, table inventory, page count, mutex state."""
    smalt_dir = str(app.cfg.smalt_dir)
    exists = app.smalt_exists()

    if not exists:
        return {
            "smalt_dir": smalt_dir,
            "exists": False,
            "tables": [],
            "page_count": 0,
            "mutex": {"locked": app.mutex.locked, "holder": app.mutex.holder},
            "embedding": {
                "provider": app.cfg.embedding.provider,
                "model": app.cfg.embedding.model,
                "dim": app.cfg.embedding.dim,
            },
        }

    tables: list[str] = []
    page_count = 0
    try:
        db = app.db()
        tables = lance.list_tables(app.cfg.smalt_dir)
        if lance.TABLE_PAGES in tables:
            page_count = db.open_table(lance.TABLE_PAGES).count_rows()
    except FileNotFoundError:
        pass  # smalt_exists() said yes but the index dir specifically is missing

    return {
        "smalt_dir": smalt_dir,
        "exists": True,
        "tables": tables,
        "page_count": page_count,
        "mutex": {"locked": app.mutex.locked, "holder": app.mutex.holder},
        "embedding": {
            "provider": app.cfg.embedding.provider,
            "model": app.cfg.embedding.model,
            "dim": app.cfg.embedding.dim,
        },
    }


# ---- handler: index_status (C-8) ----


@_wrap_sync_in_thread
def index_status(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Report detailed index + indexer state.

    Returns the same payload as the `GET /admin/health` HTTP route
    (assembled by `App.index_status_payload()`). Lets MCP clients
    introspect server health without an out-of-band HTTP roundtrip —
    useful for cogrindd's M2.5 host that holds the MCP session but
    doesn't necessarily talk HTTP to the same server.

    The payload covers:
      - Smalt path + existence
      - Per-table row counts (pages, embeddings, links, claims, sources)
      - Last indexer run metadata (timestamp, duration, full IndexResult)
      - Per-index build status (FTS per-field; ANN with skip-vs-fail-vs-ok)
      - Embedding config + whether the model is loaded
      - Mutex contention (locked-now, holder, acquire count, mean wait ms)

    Always safe to call (no writes, no LanceDB mutations); returns
    nulls for not-yet-known fields when called pre-bootstrap or before
    the first indexer pass.
    """
    return app.index_status_payload()


# ---- handler: list_pages ----


@_wrap_sync_in_thread
def list_pages(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List indexed pages, optionally filtered by `type` / `prefix` (LanceDB-side)
    and/or any of the property filters (`glossary`, `is_domain`, `domain`,
    `fetched_at_before`, `fetched_at_after`, `has_aliases_containing`,
    client-side post-fetch). All filters AND-composed; `limit` applies to the
    final filtered set."""
    err = _ensure_initialized(app)
    if err is not None:
        return err

    type_filter = arguments.get("type")
    prefix = arguments.get("prefix")
    limit = int(arguments.get("limit", 100))

    property_filters, err_payload = _parse_property_filters(arguments)
    if err_payload is not None:
        return err_payload
    has_props = bool(property_filters)

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)

    where_parts: list[str] = []
    if type_filter:
        where_parts.append(f"type = {lance.sql_str(type_filter)}")
    if prefix:
        where_parts.append(f"id LIKE {lance.sql_str(prefix + '%')}")

    # Column selection: pull `frontmatter_json` too when any property
    # filter is set (we parse it client-side). Avoid the extra column
    # otherwise — slightly cheaper on big Smalts.
    select_cols = ["id", "title", "type", "path"]
    if has_props:
        select_cols.append("frontmatter_json")

    query = pages.search().select(select_cols)
    if where_parts:
        query = query.where(" AND ".join(where_parts))

    # When property filters are set, we don't know how many SQL-filtered
    # rows will survive client-side filtering, so over-fetch and trim.
    # Heuristic: fetch up to 10x the requested limit (capped at 10_000)
    # when filters are present; otherwise fetch exactly `limit`. If the
    # over-fetch budget is exhausted the response sets `truncated: true`.
    fetch_limit = limit if not has_props else max(limit * 10, 1000)
    fetch_limit = min(fetch_limit, 10_000)
    arrow = query.limit(fetch_limit).to_arrow()

    ids = arrow.column("id").to_pylist()
    titles = arrow.column("title").to_pylist()
    types_ = arrow.column("type").to_pylist()
    paths = arrow.column("path").to_pylist()
    fms_raw = arrow.column("frontmatter_json").to_pylist() if has_props else [None] * len(ids)

    out_pages: list[dict[str, Any]] = []
    for pid, t, tp, p, fm_raw in zip(ids, titles, types_, paths, fms_raw, strict=True):
        if has_props:
            try:
                fm = json.loads(fm_raw) if fm_raw else {}
            except json.JSONDecodeError:
                continue
            if not _apply_property_filters(fm, property_filters):
                continue
        out_pages.append({"id": pid, "title": t, "type": tp, "path": p})
        if len(out_pages) >= limit:
            break

    truncated = has_props and arrow.num_rows >= fetch_limit and len(out_pages) < limit
    return {"pages": out_pages, "count": len(out_pages), "truncated": truncated}


# ---- alias-lookup helpers (shared by find_by_alias + read_page fallback) ----


def _find_pages_by_alias(app: App, alias: str) -> list[dict[str, Any]]:
    """Return every indexed page whose `aliases` list contains `alias`.

    Uses the first-class `pages.aliases` list-of-string column via
    LanceDB's `array_has` SQL predicate — O(1) at the LanceDB layer
    rather than the O(N) frontmatter_json scan we used pre-C-10.

    On a pre-C-10 Smalt (column-missing or NULL for un-reprojected
    rows), `array_has(NULL, ...)` returns NULL → those rows simply
    don't match. Operators run `reindex_all` (C-9) to populate the
    column for legacy rows, or the column fills in progressively as
    pages get re-projected by natural rewrites.
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    if "aliases" not in pages.schema.names:
        # Pre-C-10 Smalt where the additive migration didn't run. Bail
        # back to the legacy scan path so existing callers don't break
        # during upgrade. Operators should call ensure_tables /
        # reindex_all to migrate, after which this branch is dead.
        return _find_pages_by_alias_legacy_scan(app, alias)
    arrow = (
        pages.search()
        .where(f"array_has(aliases, {lance.sql_str(alias)})")
        .select(["id", "title", "type", "path"])
        .limit(10_000)
        .to_arrow()
    )
    matches: list[dict[str, Any]] = []
    for i in range(arrow.num_rows):
        matches.append(
            {
                "id": arrow.column("id")[i].as_py(),
                "title": arrow.column("title")[i].as_py(),
                "type": arrow.column("type")[i].as_py(),
                "path": arrow.column("path")[i].as_py(),
            }
        )
    return matches


def _find_pages_by_alias_legacy_scan(app: App, alias: str) -> list[dict[str, Any]]:
    """Pre-C-10 frontmatter_json scan path. Kept as a defensive fallback
    for Smalts whose pages table predates the aliases-column migration
    (the migration runs on every ensure_tables, so this should only fire
    if migration was skipped or failed). Same shape as
    `_find_pages_by_alias`."""
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = pages.search().select(["id", "title", "type", "path", "frontmatter_json"]).to_arrow()
    matches: list[dict[str, Any]] = []
    for i in range(arrow.num_rows):
        fm_raw = arrow.column("frontmatter_json")[i].as_py()
        try:
            fm = json.loads(fm_raw) if fm_raw else {}
        except json.JSONDecodeError:
            continue
        aliases = fm.get("aliases") or []
        if alias in aliases:
            matches.append(
                {
                    "id": arrow.column("id")[i].as_py(),
                    "title": arrow.column("title")[i].as_py(),
                    "type": arrow.column("type")[i].as_py(),
                    "path": arrow.column("path")[i].as_py(),
                }
            )
    return matches


def _fetch_page_row(app: App, canonical_id: str) -> dict[str, Any] | None:
    """Return the full read_page payload for `canonical_id`, or None if not indexed."""
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = (
        pages.search()
        .where(f"id = {lance.sql_str(canonical_id)}")
        .select(["id", "title", "type", "path", "body", "frontmatter_json"])
        .limit(1)
        .to_arrow()
    )
    if arrow.num_rows == 0:
        return None
    fm_raw = arrow.column("frontmatter_json")[0].as_py()
    try:
        fm = json.loads(fm_raw) if fm_raw else {}
    except json.JSONDecodeError:
        fm = {}
    return {
        "id": arrow.column("id")[0].as_py(),
        "title": arrow.column("title")[0].as_py(),
        "type": arrow.column("type")[0].as_py(),
        "path": arrow.column("path")[0].as_py(),
        "body": arrow.column("body")[0].as_py(),
        "frontmatter": fm,
    }


# ---- handler: read_page ----


@_wrap_sync_in_thread
def read_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the frontmatter (parsed) + body of a single page.

    Lookup order:
      1. Exact match on the canonical `id`. If found, return.
      2. Exact alias fallback: search every page's `aliases` for `page_id`.
         - Exactly one match → return that page, with `resolved_via_alias`.
         - Two or more matches → `{error: 'ambiguous_alias', matches: [...]}`.
         - Zero matches → fall through to step 3.
      3. Fuzzy alias fallback (C-11; opt-out via `fuzzy=false`): trigram-
         Jaccard match `page_id` against every page's aliases.
         - Exactly one match → return that page, with
           `resolved_via_alias` + `fuzzy: true` + `fuzzy_score` +
           `matched_alias`.
         - Two or more matches → `{error: 'ambiguous_alias',
           matches: [...], fuzzy: true}`.
         - Zero matches → `{error: 'not_found', page_id: ..., fuzzy: true}`
           (the `fuzzy: true` here signals we tried but didn't find).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    fuzzy = arguments.get("fuzzy", True)

    # 1. Exact id match
    payload = _fetch_page_row(app, page_id)
    if payload is not None:
        return payload

    # 2. Exact alias fallback
    matches = _find_pages_by_alias(app, page_id)
    if matches:
        if len(matches) > 1:
            return {
                "error": "ambiguous_alias",
                "alias": page_id,
                "matches": matches,
                "fuzzy": False,
                "message": (
                    f"alias {page_id!r} matches {len(matches)} pages; "
                    "address by canonical id (use the `id` field of one of the matches above)"
                ),
            }
        # Exactly one exact match — fetch the full row.
        canonical = matches[0]["id"]
        payload = _fetch_page_row(app, canonical)
        if payload is None:  # shouldn't happen — index just told us this exists
            return {"error": "not_found", "page_id": canonical}
        payload["resolved_via_alias"] = page_id
        return payload

    # 3. Fuzzy alias fallback (opt-out via fuzzy=false).
    if not fuzzy:
        return {"error": "not_found", "page_id": page_id}
    fuzzy_matches = _find_pages_by_alias_fuzzy(app, page_id)
    if not fuzzy_matches:
        return {
            "error": "not_found",
            "page_id": page_id,
            "fuzzy": True,
            "fuzzy_threshold": _fuzzy_alias_threshold(),
        }
    if len(fuzzy_matches) > 1:
        return {
            "error": "ambiguous_alias",
            "alias": page_id,
            "matches": fuzzy_matches,
            "fuzzy": True,
            "fuzzy_threshold": _fuzzy_alias_threshold(),
            "message": (
                f"alias {page_id!r} fuzzy-matches {len(fuzzy_matches)} pages "
                f"at threshold {_fuzzy_alias_threshold()}; address by canonical id "
                "(use the `id` field of one of the matches above)"
            ),
        }
    canonical = fuzzy_matches[0]["id"]
    payload = _fetch_page_row(app, canonical)
    if payload is None:  # shouldn't happen — index just told us this exists
        return {"error": "not_found", "page_id": canonical}
    payload["resolved_via_alias"] = page_id
    payload["fuzzy"] = True
    payload["fuzzy_score"] = fuzzy_matches[0]["fuzzy_score"]
    payload["matched_alias"] = fuzzy_matches[0]["matched_alias"]
    return payload


# ---- handler: find_by_alias ----


@_wrap_sync_in_thread
def find_by_alias(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List every page whose `aliases` contains `alias`.

    Use this when you have a memorable handle (the original caller-id
    before mangling, or any hand-added alias) and want to find the page(s)
    it maps to. Returns minimal metadata (id, title, type, path) per match
    — call `read_page` with the canonical `id` to get the body.

    Resolution order (C-11):
      1. Exact alias match. If any matches, return them — `fuzzy: false`.
      2. If `fuzzy` arg is true (default) and step 1 returned zero
         matches, fall back to trigram-Jaccard fuzzy match. Returned rows
         include `fuzzy_score` (Jaccard sim ∈ [threshold, 1.0]) and
         `matched_alias` (the alias string that scored highest for that
         page). Top-level `fuzzy: true` flags the fallback fired.
      3. If still zero, return `count: 0` (no error).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    alias = arguments.get("alias")
    if not alias:
        return {"error": "missing_argument", "message": "alias is required"}
    fuzzy = arguments.get("fuzzy", True)

    matches = _find_pages_by_alias(app, alias)
    if matches:
        return {
            "alias": alias,
            "matches": matches,
            "count": len(matches),
            "fuzzy": False,
        }
    if not fuzzy:
        return {"alias": alias, "matches": [], "count": 0, "fuzzy": False}
    fuzzy_matches = _find_pages_by_alias_fuzzy(app, alias)
    return {
        "alias": alias,
        "matches": fuzzy_matches,
        "count": len(fuzzy_matches),
        "fuzzy": bool(fuzzy_matches),
        "fuzzy_threshold": _fuzzy_alias_threshold(),
    }


# ---- handler: traverse ----

# Hard cap on hops. Set high enough that real Cogitate observer walks fit
# (entity-cluster detection typically wants 2-3 hops); low enough that a
# pathologically dense graph can't blow up the response. Configurable via the
# call (`hops` arg) up to this ceiling; calls above this ceiling return an
# `invalid_argument` error rather than silently clamping.
_MAX_TRAVERSE_HOPS = 5

# Per-hop edge cap. The single-hop v0 limit was 1000; same limit per BFS
# level for multi-hop. If a single hop's outgoing-edge set exceeds this,
# the response sets `truncated: true` so the caller knows the walk was
# incomplete.
_PER_HOP_EDGE_LIMIT = 1000


@_wrap_sync_in_thread
def traverse(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Multi-hop outgoing-link graph traversal via BFS.

    Returns the union of outgoing edges discovered within `hops` hops of
    `from_id`, plus the set of nodes visited (including `from_id` as the
    seed). Optional `label` filter is applied **per hop** — only edges with
    that label are followed; nodes reachable only via other labels don't
    expand.

    Cycle handling: revisited nodes don't re-expand (BFS visited-set).
    Self-loops are collected as edges but don't re-expand.
    `hops` defaults to 1; max is `_MAX_TRAVERSE_HOPS` (5); calls above the
    ceiling get `invalid_argument`.

    Each hop's outgoing-edge query is capped at `_PER_HOP_EDGE_LIMIT` (1000)
    rows; if a hop hits the cap, the response sets `truncated: true`.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    from_id = arguments.get("from_id")
    if not from_id:
        return {"error": "missing_argument", "message": "from_id is required"}
    label = arguments.get("label")
    hops = int(arguments.get("hops", 1))
    if hops < 1:
        return {
            "error": "invalid_argument",
            "message": f"hops must be >= 1; got {hops}",
        }
    if hops > _MAX_TRAVERSE_HOPS:
        return {
            "error": "invalid_argument",
            "message": (
                f"hops must be <= {_MAX_TRAVERSE_HOPS}; got {hops}. "
                "Higher hop counts risk huge result sets on dense graphs; "
                "raise the cap or paginate client-side if your case really needs it."
            ),
        }

    db = app.db()
    links_table = db.open_table(lance.TABLE_LINKS)

    visited: set[str] = {from_id}
    edges: list[dict[str, Any]] = []
    frontier: list[str] = [from_id]
    truncated = False

    for _hop in range(hops):
        if not frontier:
            break

        # Build WHERE clause: from_id IN (...) [AND label = ?]. The IN-list
        # syntax mirrors the search handler's hit-set hydration query (see
        # search handler — `quoted = ", ".join(lance.sql_str(p) ...)`).
        quoted_ids = ", ".join(lance.sql_str(node) for node in frontier)
        where = f"from_id IN ({quoted_ids})"
        if label:
            where += f" AND label = {lance.sql_str(label)}"

        arrow = (
            links_table.search()
            .where(where)
            .select(["from_id", "to_id", "label"])
            .limit(_PER_HOP_EDGE_LIMIT)
            .to_arrow()
        )

        n_rows = arrow.num_rows
        if n_rows >= _PER_HOP_EDGE_LIMIT:
            truncated = True

        from_ids = arrow.column("from_id").to_pylist()
        to_ids = arrow.column("to_id").to_pylist()
        labels = arrow.column("label").to_pylist()

        next_frontier: list[str] = []
        for f, t, lbl in zip(from_ids, to_ids, labels, strict=True):
            edges.append({"from_id": f, "to_id": t, "label": lbl})
            if t not in visited:
                visited.add(t)
                next_frontier.append(t)
        frontier = next_frontier

    return {
        "from_id": from_id,
        "hops": hops,
        "edges": edges,
        "count": len(edges),
        "visited_nodes": sorted(visited),
        "truncated": truncated,
    }


# ---- handler: incoming_links (READ_ONLY) ----


@_wrap_sync_in_thread
def incoming_links(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List every link whose `to_id` matches `page_id` — the "what points
    at me" view.

    Symmetric to `traverse` (which lists OUTGOING links). Useful to audit
    references before calling `remove_page` (which cascades — removing a
    page silently drops all incoming references; this tool lets the caller
    see them first).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    label = arguments.get("label")

    db = app.db()
    links = db.open_table(lance.TABLE_LINKS)
    where = f"to_id = {lance.sql_str(page_id)}"
    if label:
        where += f" AND label = {lance.sql_str(label)}"
    arrow = (
        links.search()
        .where(where)
        .select(["from_id", "to_id", "label", "source_page"])
        .limit(10_000)
        .to_arrow()
    )
    edges = [
        {
            "from_id": f,
            "to_id": t,
            "label": lbl,
            "source_page": sp,
        }
        for f, t, lbl, sp in zip(
            arrow.column("from_id").to_pylist(),
            arrow.column("to_id").to_pylist(),
            arrow.column("label").to_pylist(),
            arrow.column("source_page").to_pylist(),
            strict=True,
        )
    ]
    return {"to_id": page_id, "edges": edges, "count": len(edges)}


# ---- handler: search ----


def _rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of N ranked lists. Returns (id, score) sorted desc."""
    scores: dict[str, float] = {}
    for ranks in rankings:
        for rank, pid in enumerate(ranks, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _find_alias_matches(app: App, query: str) -> list[str]:
    """Return page_ids whose aliases match the query string.

    Match rule: the page matches if **the whole query** appears verbatim
    in its aliases list, OR if any whitespace-separated token of the
    query appears verbatim. This handles both the common "search for the
    alias directly" case (`search('ent-alice')`) and the embedded case
    (`search('tell me about ent-alice')`).

    Implementation (C-10): builds a single SQL expression of OR'd
    `array_has(aliases, <needle>)` predicates, one per distinct needle
    (the whole-query string + each whitespace-separated token). The
    LanceDB query engine evaluates this in one pass over the indexed
    column — O(1) per row at the engine level rather than the O(N)
    frontmatter_json scan we used pre-C-10.

    Returns page_ids in `pages`-table-scan order. The downstream RRF
    treats them as a single ranked list — pages that match earlier in
    the scan get slightly higher rank, but the ranking signal is weak.
    The point is presence in the input set, not relative ordering.
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    if "aliases" not in pages.schema.names:
        # Legacy scan fallback for pre-C-10 Smalts (see
        # _find_pages_by_alias for the same defensive pattern).
        return _find_alias_matches_legacy_scan(app, query)

    needles: set[str] = {query}
    needles.update(t.strip() for t in query.split() if t.strip())
    if not needles:
        return []
    # OR together one array_has predicate per needle.
    where = " OR ".join(f"array_has(aliases, {lance.sql_str(n)})" for n in needles)
    arrow = pages.search().where(where).select(["id"]).limit(10_000).to_arrow()
    return arrow.column("id").to_pylist()


def _find_alias_matches_legacy_scan(app: App, query: str) -> list[str]:
    """Pre-C-10 frontmatter_json scan fallback. See
    `_find_pages_by_alias_legacy_scan` for the rationale."""
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = pages.search().select(["id", "frontmatter_json"]).to_arrow()
    needles: set[str] = {query}
    needles.update(t.strip() for t in query.split() if t.strip())
    matches: list[str] = []
    for i in range(arrow.num_rows):
        fm_raw = arrow.column("frontmatter_json")[i].as_py()
        try:
            fm = json.loads(fm_raw) if fm_raw else {}
        except json.JSONDecodeError:
            continue
        aliases = set(fm.get("aliases") or [])
        if aliases & needles:
            matches.append(arrow.column("id")[i].as_py())
    return matches


# ---- fuzzy alias match (C-11) ----
#
# Trigram-set Jaccard similarity over the alias strings. Used as a
# fallback by `find_by_alias` and `read_page` when exact match returns
# zero hits. Search retrieval (`_find_alias_matches` above) stays
# exact-only — fuzzy noise in the RRF set would degrade ranking quality
# more than it'd help with typo tolerance.
#
# Why trigrams + Jaccard? Slug-shape aliases ("ent-alice",
# "domain-cogitate-methodology") are short and structured; trigrams
# capture local character order well, Jaccard penalizes length mismatch
# heavily so unrelated short strings don't drag in spuriously. The
# tradeoff vs. Levenshtein: Jaccard is set-based (no edit-distance
# matrix), evaluates in O(|a| + |b|) per pair, and scores ~equivalently
# on the typo / near-miss cases we care about.
#
# Threshold: 0.6 default. "ent-alic" vs "ent-alice" ≈ 0.857 (passes);
# "ent-alice" vs "ent-bob" ≈ 0.2 (correctly excluded); "ent-alice" vs
# "ent-alicia" ≈ 0.667 (passes — both spellings near-match). Tightening
# above ~0.7 starts missing common typos; loosening below ~0.5 starts
# dragging in unrelated slugs. Override via env for ops tuning.
_FUZZY_ALIAS_THRESHOLD_ENV = "SMALT_FUZZY_ALIAS_THRESHOLD"
_DEFAULT_FUZZY_ALIAS_THRESHOLD = 0.6


def _fuzzy_alias_threshold() -> float:
    """Read the fuzzy-match threshold from env, falling back to the default.

    Invalid env values (non-numeric, out of `(0, 1]`) log a warning and
    fall back to default so a typo in deployment config doesn't silently
    disable the fallback.
    """
    raw = os.environ.get(_FUZZY_ALIAS_THRESHOLD_ENV)
    if raw is None or raw == "":
        return _DEFAULT_FUZZY_ALIAS_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not numeric; using default %s",
            _FUZZY_ALIAS_THRESHOLD_ENV,
            raw,
            _DEFAULT_FUZZY_ALIAS_THRESHOLD,
        )
        return _DEFAULT_FUZZY_ALIAS_THRESHOLD
    if not (0.0 < v <= 1.0):
        logger.warning(
            "%s=%r is out of range (need float in (0, 1]); using default %s",
            _FUZZY_ALIAS_THRESHOLD_ENV,
            raw,
            _DEFAULT_FUZZY_ALIAS_THRESHOLD,
        )
        return _DEFAULT_FUZZY_ALIAS_THRESHOLD
    return v


def _trigram_set(s: str) -> set[str]:
    """Char-trigram (3-gram) set of `s`, lower-cased.

    Returns empty for strings shorter than 3 chars (caller treats that
    as "can't fuzzy match — must be exact"). No padding chars: slug
    aliases are short enough that boundary padding would dominate the
    intersection and inflate scores misleadingly.
    """
    s = s.lower()
    if len(s) < 3:
        return set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _jaccard_trigram(a: str, b: str) -> float:
    """Trigram-set Jaccard similarity: |A intersect B| / |A union B|.

    Returns 0.0 if either operand's trigram set is empty (string < 3
    chars) or if both are empty (so the union is empty too).
    """
    sa = _trigram_set(a)
    sb = _trigram_set(b)
    if not sa or not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _find_pages_by_alias_fuzzy(
    app: App, query: str, *, threshold: float | None = None
) -> list[dict[str, Any]]:
    """Fuzzy-match `query` against every page's aliases via trigram Jaccard.

    Returns matches sorted by similarity (highest first). Each row carries
    the standard `{id, title, type, path}` shape plus:
      - `fuzzy_score`: float in [threshold, 1.0], the alias's best score
      - `matched_alias`: the alias string that scored highest for this page

    Implementation note: O(N pages x M aliases per page) score evaluations.
    Acceptable at hundreds-to-low-thousands of pages — at our current
    scale a 1k-page Smalt with ~3 aliases per page scores in ms. If this
    becomes a bottleneck, materialize a trigram inverted index (separate
    Lance table keyed by trigram); defer that until a real-world Smalt
    is large enough to feel it.

    Tie-breaking: rows with equal `fuzzy_score` ordered by `id`
    (deterministic; doesn't depend on scan order, so callers see a
    stable list across runs).
    """
    threshold = threshold if threshold is not None else _fuzzy_alias_threshold()
    query_set = _trigram_set(query)
    if not query_set:
        return []

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    has_col = "aliases" in pages.schema.names
    cols = ["id", "title", "type", "path"]
    if has_col:
        cols.append("aliases")
    else:
        cols.append("frontmatter_json")
    arrow = pages.search().select(cols).limit(10_000).to_arrow()

    results: list[tuple[float, str, dict[str, Any]]] = []
    for i in range(arrow.num_rows):
        if has_col:
            aliases = arrow.column("aliases")[i].as_py() or []
        else:
            fm_raw = arrow.column("frontmatter_json")[i].as_py()
            try:
                fm = json.loads(fm_raw) if fm_raw else {}
            except json.JSONDecodeError:
                continue
            aliases = fm.get("aliases") or []
        if not aliases:
            continue
        best_score = 0.0
        best_alias: str | None = None
        for alias in aliases:
            score = _jaccard_trigram(query, alias)
            if score > best_score:
                best_score = score
                best_alias = alias
        if best_score >= threshold and best_alias is not None:
            pid = arrow.column("id")[i].as_py()
            row = {
                "id": pid,
                "title": arrow.column("title")[i].as_py(),
                "type": arrow.column("type")[i].as_py(),
                "path": arrow.column("path")[i].as_py(),
                "fuzzy_score": round(best_score, 4),
                "matched_alias": best_alias,
            }
            results.append((best_score, pid, row))

    # Sort by score desc, then id asc for deterministic ordering on ties.
    results.sort(key=lambda triple: (-triple[0], triple[1]))
    return [row for _, _, row in results]


@_wrap_sync_in_thread
def search(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Hybrid search over the pages corpus: FTS (body) + vector (embeddings) +
    alias retrieval, RRF-fused, with optional property filters applied
    post-fusion before the top_k cap.

    Returns the top-`top_k` matches with id, title, type, snippet, score,
    aliases. If the FTS index hasn't been built yet (very small Smalt),
    falls back to vector-only ranking.

    Property filters (`glossary`, `is_domain`, `domain`, `fetched_at_before`,
    `fetched_at_after`, `has_aliases_containing`) are applied to the hydrated
    candidates AFTER RRF fusion and BEFORE top_k truncation, so the top_k
    you get back is "top_k matches that pass the filter," not "top_k matches
    of which some happen to pass." When filters are set, the retrieval
    over-fetches further to keep the post-filter candidate pool deep.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    query = arguments.get("query")
    if not query:
        return {"error": "missing_argument", "message": "query is required"}
    top_k = int(arguments.get("top_k", 10))

    property_filters, err_payload = _parse_property_filters(arguments)
    if err_payload is not None:
        return err_payload
    has_props = bool(property_filters)

    # Over-fetch for RRF; when filters are set, over-fetch further so the
    # post-filter pool stays deep enough to fill top_k.
    fetch_k = max(top_k * 3, top_k + 5)
    if has_props:
        fetch_k = max(fetch_k * 5, 100)  # 5x extra room for filter dropoff

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)

    # FTS over body (and title; LanceDB FTS indexes one field per call, we
    # have one for each — search uses whichever the table considers primary).
    fts_ids: list[str] = []
    try:
        fts_arrow = pages.search(query, query_type="fts").select(["id"]).limit(fetch_k).to_arrow()
        fts_ids = fts_arrow.column("id").to_pylist()
    except Exception as e:
        logger.info("FTS search unavailable (%s); falling back to vector-only", e)

    # Vector over embeddings table; pull page_ids ranked by similarity.
    vec = app.embedder().embed([query])[0]
    embs = db.open_table(lance.TABLE_EMBEDDINGS)
    try:
        vec_arrow = (
            embs.search(vec, vector_column_name="vector").select(["page_id"]).limit(fetch_k).to_arrow()
        )
        vec_ids = vec_arrow.column("page_id").to_pylist()
    except Exception as e:
        logger.warning("vector search failed: %s", e)
        vec_ids = []

    # Third retrieval source: aliases. Catches the case where the query is
    # (or contains) a page's alias — e.g., searching for the caller-id of a
    # mangled page. FTS won't surface this because aliases aren't in the
    # FTS-indexed columns; vector similarity won't either because the alias
    # string isn't in the page body.
    alias_ids = _find_alias_matches(app, query)

    if not fts_ids and not vec_ids and not alias_ids:
        return {"results": [], "count": 0}

    fused = _rrf_fuse([fts_ids, vec_ids, alias_ids])
    # When property filters are set, hydrate the FULL fused candidate
    # pool (not just top_k) so we have enough rows to apply the filter
    # and still fill top_k. Cap the hydration set at fetch_k to bound the
    # IN-clause + parse cost.
    candidate_pool = fused[:fetch_k] if has_props else fused[:top_k]
    candidate_ids = [pid for pid, _ in candidate_pool]

    if not candidate_ids:
        return {"results": [], "count": 0}

    # Hydrate with page metadata in one query. Pull frontmatter_json too so we
    # can (a) surface aliases per hit — callers often want to render results
    # by a memorable handle (the caller-id-now-alias) rather than the
    # canonical id — and (b) apply property filters client-side.
    quoted = ", ".join(lance.sql_str(p) for p in candidate_ids)
    meta_arrow = (
        pages.search()
        .where(f"id IN ({quoted})")
        .select(["id", "title", "type", "body", "frontmatter_json"])
        .limit(len(candidate_ids))
        .to_arrow()
    )
    by_id: dict[str, dict[str, Any]] = {}
    for i in range(meta_arrow.num_rows):
        pid = meta_arrow.column("id")[i].as_py()
        fm_raw = meta_arrow.column("frontmatter_json")[i].as_py()
        try:
            fm = json.loads(fm_raw) if fm_raw else {}
        except json.JSONDecodeError:
            fm = {}
        if has_props and not _apply_property_filters(fm, property_filters):
            continue
        aliases = list(fm.get("aliases") or [])
        by_id[pid] = {
            "title": meta_arrow.column("title")[i].as_py(),
            "type": meta_arrow.column("type")[i].as_py(),
            "body": meta_arrow.column("body")[i].as_py() or "",
            "aliases": aliases,
        }

    # Re-rank: walk the fused (RRF-ordered) candidate pool, picking up
    # filter-passing hits until top_k or the pool runs out.
    results: list[dict[str, Any]] = []
    for pid, score in candidate_pool:
        meta = by_id.get(pid)
        if not meta:
            continue
        body = meta["body"]
        snippet = body[:200] + ("…" if len(body) > 200 else "")
        results.append(
            {
                "id": pid,
                "aliases": meta["aliases"],
                "title": meta["title"],
                "type": meta["type"],
                "snippet": snippet,
                "score": round(score, 6),
            }
        )
        if len(results) >= top_k:
            break

    return {"results": results, "count": len(results)}


# ---- handler: list_domains (READ_ONLY) ----


@_wrap_sync_in_thread
def list_domains(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List ConceptPages flagged `is_domain: true`.

    Domain hierarchy itself (which domain is a subdomain of which) lives in
    each domain's `subdomain_of` labeled links in `links_out`, NOT in this
    response. Use `traverse(from_id=<domain>, label='subdomain_of')` to walk
    the hierarchy.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = (
        pages.search()
        .where(f"type = {lance.sql_str(PageType.CONCEPT.value)}")
        .select(["id", "title", "path", "frontmatter_json"])
        .limit(10_000)
        .to_arrow()
    )
    domains: list[dict[str, Any]] = []
    for i in range(arrow.num_rows):
        fm_raw = arrow.column("frontmatter_json")[i].as_py()
        try:
            fm = json.loads(fm_raw) if fm_raw else {}
        except json.JSONDecodeError:
            continue
        if fm.get("is_domain") is not True:
            continue
        domains.append(
            {
                "id": arrow.column("id")[i].as_py(),
                "title": arrow.column("title")[i].as_py(),
                "path": arrow.column("path")[i].as_py(),
            }
        )
    domains.sort(key=lambda d: d["id"])
    return {"domains": domains, "count": len(domains)}


# ---- handler: source_similarity (READ_ONLY, C-12) ----
#
# Vector search using a page's stored embedding as the query vector —
# "what other pages are most similar to this one?" — without paying
# the embed cost of a new query string. The arg is conventionally
# named `source_id` because the page sources the query vector, NOT
# because the input must be a SourcePage; any indexed page works.
#
# Cosine similarity, derived from LanceDB's `_distance` column
# (built with metric="cosine"). similarity = 1 - distance. Range
# nominally [-1, 1]; with normalized embeddings in practice [0, 1].
#
# Type filter: optional list of page-type strings (e.g.
# `types: ["source"]`) restricts results to those types. Filtering
# happens client-side after the vector hit list comes back from
# LanceDB (over-fetched 5x to preserve depth post-filter); the
# `embeddings` table doesn't carry a type column.


@_wrap_sync_in_thread
def source_similarity(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Pages most similar to `source_id`'s embedding by cosine similarity.

    Excludes the source page itself from the result list. Returns up to
    `top_k` matches (default 10) with id, title, type, path, aliases,
    and `similarity` (= 1 - cosine distance, range nominally [-1, 1];
    with normalized embeddings, practically [0, 1] where 1.0 = identical).

    Optional `types` arg (list of page-type strings) filters the result
    set. Validated against the known PageType enum; unknown types return
    `invalid_argument` rather than silently producing zero results.

    Errors:
      - `missing_argument` if `source_id` is omitted.
      - `invalid_argument` if `top_k <= 0` or `types` is malformed.
      - `not_found` if `source_id` has no embedding (page not indexed
        or embedding generation failed — caller should try
        `reindex_page` first).
      - `vector_search_failed` if LanceDB raises mid-search (rare;
        usually means the ANN index is in a bad state).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    source_id = arguments.get("source_id")
    if not source_id:
        return {"error": "missing_argument", "message": "source_id is required"}
    try:
        top_k = int(arguments.get("top_k", 10))
    except (TypeError, ValueError):
        return {
            "error": "invalid_argument",
            "message": "top_k must be an integer",
        }
    if top_k <= 0:
        return {
            "error": "invalid_argument",
            "message": f"top_k must be positive (got {top_k})",
        }
    types = arguments.get("types")
    if types is not None:
        if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
            return {
                "error": "invalid_argument",
                "message": "types must be a list of page-type strings",
            }
        valid_types = {p.value for p in PageType}
        unknown = [t for t in types if t not in valid_types]
        if unknown:
            return {
                "error": "invalid_argument",
                "message": (f"unknown page type(s): {unknown!r}; valid: {sorted(valid_types)}"),
            }

    db = app.db()
    embs = db.open_table(lance.TABLE_EMBEDDINGS)

    # 1. Fetch the source's stored embedding.
    src_arrow = (
        embs.search()
        .where(f"page_id = {lance.sql_str(source_id)}")
        .select(["page_id", "vector"])
        .limit(1)
        .to_arrow()
    )
    if src_arrow.num_rows == 0:
        return {
            "error": "not_found",
            "source_id": source_id,
            "message": (
                f"no embedding for page {source_id!r} — page may not be "
                "indexed, or embedding generation failed; try reindex_page"
            ),
        }
    query_vec = src_arrow.column("vector")[0].as_py()

    # 2. Over-fetch by 5x when filtering by type so the post-filter pool
    # stays deep enough to fill top_k. +1 to cover the self-exclusion. The
    # `page_id != source_id` predicate runs at the LanceDB layer so the
    # source row never even appears in the hit list — `+1` is here for
    # safety, not correctness.
    fetch_k = top_k + 1
    if types is not None:
        fetch_k = max(fetch_k * 5, 50)

    try:
        hit_arrow = (
            embs.search(query_vec, vector_column_name="vector")
            .where(f"page_id != {lance.sql_str(source_id)}")
            .select(["page_id", "_distance"])
            .limit(fetch_k)
            .to_arrow()
        )
    except Exception as e:
        return {
            "error": "vector_search_failed",
            "source_id": source_id,
            "message": f"vector search raised: {e}",
        }

    if hit_arrow.num_rows == 0:
        return {
            "source_id": source_id,
            "results": [],
            "count": 0,
            "types_filter": types,
        }

    hit_ids = hit_arrow.column("page_id").to_pylist()
    hit_distances = hit_arrow.column("_distance").to_pylist()
    id_to_distance = dict(zip(hit_ids, hit_distances, strict=False))

    # 3. Hydrate hit_ids with page metadata in one query. We pull the
    # frontmatter_json so we can surface `aliases` per row (consistent
    # with the `search` handler's response shape).
    pages = db.open_table(lance.TABLE_PAGES)
    quoted = ", ".join(lance.sql_str(p) for p in hit_ids)
    meta_arrow = (
        pages.search()
        .where(f"id IN ({quoted})")
        .select(["id", "title", "type", "path", "frontmatter_json"])
        .limit(len(hit_ids))
        .to_arrow()
    )
    meta_by_id: dict[str, dict[str, Any]] = {}
    for i in range(meta_arrow.num_rows):
        pid = meta_arrow.column("id")[i].as_py()
        fm_raw = meta_arrow.column("frontmatter_json")[i].as_py()
        try:
            fm = json.loads(fm_raw) if fm_raw else {}
        except json.JSONDecodeError:
            fm = {}
        meta_by_id[pid] = {
            "title": meta_arrow.column("title")[i].as_py(),
            "type": meta_arrow.column("type")[i].as_py(),
            "path": meta_arrow.column("path")[i].as_py(),
            "aliases": list(fm.get("aliases") or []),
        }

    # 4. Walk hit_ids in vector-search rank order (already cosine-sorted
    # by LanceDB), applying the optional type filter, until `top_k` results
    # accumulate or the pool is exhausted.
    # `types` was validated as a list[str] above (the `all(isinstance(t, str))`
    # guard); cast restores that element type for the type checker.
    type_set: set[str] | None = set(cast("list[str]", types)) if types else None
    results: list[dict[str, Any]] = []
    for pid in hit_ids:
        meta = meta_by_id.get(pid)
        if meta is None:
            # Embedding row exists but pages row is missing (shouldn't
            # normally happen — the indexer projects both in one pass).
            # Skip rather than break: a stale embedding row is a bug
            # surface for index_status, not source_similarity.
            continue
        if type_set is not None and meta["type"] not in type_set:
            continue
        distance = id_to_distance.get(pid, 1.0)
        similarity = 1.0 - distance
        results.append(
            {
                "id": pid,
                "title": meta["title"],
                "type": meta["type"],
                "path": meta["path"],
                "aliases": meta["aliases"],
                "similarity": round(similarity, 6),
            }
        )
        if len(results) >= top_k:
            break

    truncated = types is not None and len(results) < top_k and hit_arrow.num_rows >= fetch_k
    return {
        "source_id": source_id,
        "results": results,
        "count": len(results),
        "types_filter": types,
        "truncated": truncated,
    }


# ---- handler: task_status, task_list (READ_ONLY, C-13) ----
#
# These tools surface the in-process Scheduler's task registry —
# what's queued, what's running, what just finished. They're
# READ_ONLY because they don't touch the corpus; the cancel tool
# (task_cancel) is in READ_WRITE because it affects ongoing
# corpus-modifying work.
#
# In-memory registry → tasks vanish after `gc_ttl_seconds` past
# finished_at. Operators who care about historical task records
# should poll on a faster cadence than the TTL.


@_wrap_sync_in_thread
def task_status(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the current state of one scheduled task by `task_id`.

    Tasks transition pending → running → terminal (succeeded /
    failed / cancelled). Once terminal, the task stays in the
    registry until GC removes it (`gc_ttl_seconds` after
    `finished_at`; default 1 hour).

    Response shape: the full `Task.to_dict()` — `task_id`, `kind`,
    `state`, timestamps (`created_at` / `started_at` / `finished_at`),
    `progress` (kind-specific dict), `result` (the work function's
    return value, populated only on `succeeded`), `error` (short
    string, populated only on `failed`), `cancel_requested` bool.

    Errors: `not_found` if `task_id` is unknown — could mean the id
    never existed or the task was GC'd.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    task_id = arguments.get("task_id")
    if not task_id:
        return {"error": "missing_argument", "message": "task_id is required"}
    task = app.scheduler.get(task_id)
    if task is None:
        return {
            "error": "not_found",
            "task_id": task_id,
            "message": (
                f"no task with id {task_id!r} — either it never existed "
                f"or it was GC'd (default TTL is "
                f"{app.scheduler._gc_ttl_seconds}s past finished_at)"
            ),
        }
    return task.to_dict()


@_wrap_sync_in_thread
def task_list(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List scheduled tasks, most-recent-first, optionally filtered.

    Filters:
      - `state` — one of `pending`, `running`, `succeeded`, `failed`,
        `cancelled`. Returns tasks in that state only.
      - `kind` — operation kind (e.g. `"reindex_all"`). Returns
        tasks of that kind only.
      - `limit` — max number of tasks to return (default 100). Applied
        after filtering — `state=succeeded, limit=10` returns the 10
        most recent successful tasks, not "10 most recent of which
        some happen to be successful."

    Returns minimal-but-useful metadata per task (the full Task
    dict, same as `task_status`).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    state_raw = arguments.get("state")
    state: TaskState | None = None
    if state_raw is not None:
        try:
            state = TaskState(state_raw)
        except ValueError:
            valid = [s.value for s in TaskState]
            return {
                "error": "invalid_argument",
                "message": f"unknown state {state_raw!r}; valid: {valid}",
            }
    kind = arguments.get("kind")
    if kind is not None and not isinstance(kind, str):
        return {
            "error": "invalid_argument",
            "message": "kind must be a string",
        }
    try:
        limit = int(arguments.get("limit", 100))
    except (TypeError, ValueError):
        return {
            "error": "invalid_argument",
            "message": "limit must be an integer",
        }
    if limit <= 0:
        return {
            "error": "invalid_argument",
            "message": f"limit must be positive (got {limit})",
        }

    tasks = app.scheduler.list(state=state, kind=kind, limit=limit)
    return {
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks),
        "state_filter": state.value if state else None,
        "kind_filter": kind,
    }


# ---- handler: bootstrap (READ_WRITE) ----


@_wrap_sync_in_thread
def bootstrap(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Initialize an empty Smalt at the configured `SMALT_DIR`.

    Creates the canonical directory layout, drops in SCHEMA.md / POLICY.md
    placeholders if they're missing, creates the LanceDB tables, and runs
    one indexer pass (which materializes the canonical IndexPages —
    `pages/glossary.md` + `pages/domains.md` — even on a fresh-empty
    Smalt, with placeholder bodies that get filled in as glossary /
    is_domain concepts are added). Idempotent: existing directories /
    files / tables are left alone; the response reports only what was
    *newly* created plus the indexer run summary.
    """
    smalt_root = app.cfg.smalt_dir
    smalt_root.mkdir(parents=True, exist_ok=True)

    created_dirs: list[str] = []
    for rel in paths.ALL_DIRS:
        d = smalt_root / rel
        if not d.exists():
            d.mkdir(parents=True)
            created_dirs.append(rel)

    created_files: list[str] = []
    for rel, content in (
        ("schema/SCHEMA.md", _SCHEMA_MD_PLACEHOLDER),
        ("schema/POLICY.md", _POLICY_MD_PLACEHOLDER),
    ):
        target = smalt_root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            created_files.append(rel)

    created_tables = lance.ensure_tables(smalt_root, embedding_dim=app.cfg.embedding.dim)

    # Kick off one indexer pass so the canonical IndexPages
    # (`pages/glossary.md`, `pages/domains.md`) get created. On a
    # fresh-empty Smalt the bodies are placeholders ("no entries yet");
    # they fill in as concepts are added and subsequent indexer passes
    # run. Wrap in the corpus mutex per `_run_indexer`'s contract.
    with app.mutex.acquire("bootstrap"):
        index_result = _run_indexer(app)

    return {
        "smalt_dir": str(smalt_root),
        "created_dirs": created_dirs,
        "created_files": created_files,
        "created_tables": created_tables,
        "index_result": index_result,
    }


# ---- write mode + existence + canonical-id helpers ----
#
# Two write modes:
#   create : ALWAYS produce a new page. The caller's id contributes to the
#            slug-prefix; a 22-char URL-safe base64 UUID4 suffix gives
#            structural uniqueness. The original id is preserved in the
#            page's `aliases` list. Collision is impossible by construction.
#   update : The caller's id must match an existing canonical id exactly.
#            Fails {error: not_found} otherwise. No mangling.
#
# There's no `upsert` mode. The two semantics — "make a new thing" and
# "modify a specific existing thing" — are deliberately distinct under
# always-mangle, because the caller's id is no longer the canonical id for
# create-writes. An upsert that sometimes-mangled-sometimes-didn't would
# be ambiguous and dangerous.


_VALID_WRITE_MODES: frozenset[str] = frozenset({"create", "update"})


def _mangle_id(caller_id: str) -> str:
    """Append a 22-char URL-safe base64 UUID4 suffix to `caller_id`.

    URL-safe base64 uses [A-Za-z0-9_-], all of which pass our PAGE_ID_RE,
    so the resulting canonical id re-validates cleanly.
    """
    suffix = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")
    return f"{caller_id}-{suffix}"


def _existing_page_path(app: App, page_id: str) -> str | None:
    """Return the indexed relative path of a page, or None if not present.

    Uses the LanceDB pages table. Caller must hold the corpus mutex if it
    intends to use the result to decide a write (otherwise a concurrent
    write could change reality before the decision lands).
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = pages.search().where(f"id = {lance.sql_str(page_id)}").select(["path"]).limit(1).to_arrow()
    if arrow.num_rows == 0:
        return None
    return arrow.column("path")[0].as_py()


def _prepare_create_write(fm_in: dict[str, Any]) -> tuple[Page, dict[str, Any], str]:
    """Prepare a create-mode write: mangle id, preserve original as alias,
    re-validate, return (page, fm_out, original_id).

    Caller passes the validated `fm_in` (already through PAGE_ADAPTER once).
    We rebuild the dict with the canonical id + extended aliases, re-validate
    so the returned `page` reflects the final on-disk shape.
    """
    original_id = fm_in["id"]
    canonical_id = _mangle_id(original_id)
    fm_out: dict[str, Any] = dict(fm_in)
    fm_out["id"] = canonical_id
    aliases = list(fm_out.get("aliases") or [])
    if original_id not in aliases:
        aliases.append(original_id)
    fm_out["aliases"] = aliases
    page = PAGE_ADAPTER.validate_python(fm_out)
    return page, fm_out, original_id


# ---- handler: write_page (READ_WRITE) ----


@_wrap_sync_in_thread
def write_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Write one page (frontmatter + body) and trigger an incremental indexer pass.

    `mode='create'` (default): always produces a NEW page. The caller's id
    becomes the slug-prefix; a 22-char URL-safe base64 UUID4 suffix makes the
    canonical id structurally unique. The original id is preserved in
    `aliases`. Returns `{id: <canonical>, original_id: <caller>, ...}` —
    callers must store the canonical id to address the page later.

    `mode='update'`: requires the caller's id to be an existing canonical id
    (no mangling, no alias resolution). Fails `{error: 'not_found'}` if no
    such page is indexed. Use this to modify a specific known page.

    Path: `pages/<subdir>/<canonical-id>.md`. Atomic tmp-then-rename. Both
    write + indexer pass run inside the corpus single-writer mutex; the
    existence check (for update) is in the same critical section.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    fm_in = arguments.get("frontmatter")
    if not fm_in:
        return {"error": "missing_argument", "message": "frontmatter is required"}
    body = arguments.get("body") or ""
    mode = arguments.get("mode") or "create"
    if mode not in _VALID_WRITE_MODES:
        return {
            "error": "invalid_argument",
            "message": f"mode must be one of {sorted(_VALID_WRITE_MODES)}; got {mode!r}",
        }

    # First validation: caller's frontmatter must conform to the Page union.
    try:
        PAGE_ADAPTER.validate_python(fm_in)
    except ValidationError as e:
        return {"error": "validation_error", "message": str(e)}

    # IndexPages are auto-generated only — the indexer regenerates them at
    # the end of every run from a `stored_query`. Direct writes here would
    # be overwritten on the next indexer pass. Reject early.
    if fm_in.get("type") == PageType.INDEX.value:
        return {
            "error": "forbidden_page_type",
            "type": PageType.INDEX.value,
            "message": (
                "IndexPages are auto-generated by the indexer at the end "
                "of every run; direct writes via write_page are rejected. "
                "To customize an index, change its stored_query manually "
                "on disk and trigger a reindex."
            ),
        }

    if mode == "create":
        # Section ids (`<source-id>::<rel-path>`) take the upsert path:
        # the id encodes a real-world location (source + rel-path tuple)
        # and is canonical by construction — mangling would break that
        # identity and produce a section file at a UUID-suffixed path
        # nothing else can find. Upsert: write the page, overwriting if
        # an existing section page lives at the same id. M3 ingest's
        # natural "re-process this file" behavior.
        if "::" in fm_in["id"]:
            page = PAGE_ADAPTER.validate_python(fm_in)
            target = _page_target_path(app.cfg.smalt_dir, page)
            with app.mutex.acquire("write_page"):
                already_existed = _existing_page_path(app, page.id) is not None
                _serialize_and_write_page(target, fm_in, body)
                index_result = _run_indexer(app)
            return {
                "id": page.id,
                "path": str(target.relative_to(app.cfg.smalt_dir)),
                "type": page.type.value,
                "mode": "create",
                "mangled": False,
                "upserted": already_existed,
                "index_result": index_result,
            }

        # Slug ids: always-mangle create. Re-validate (cheap; confirms the
        # canonical id still passes _validate_id — URL-safe base64 is in
        # the allowed character set).
        try:
            page, fm_out, original_id = _prepare_create_write(fm_in)
        except ValidationError as e:
            return {"error": "validation_error", "message": str(e)}

        target = _page_target_path(app.cfg.smalt_dir, page)
        with app.mutex.acquire("write_page"):
            # Defensive: UUID4 collision is 2^-122 per pair — effectively zero
            # — but check anyway so we never silently overwrite.
            if _existing_page_path(app, page.id) is not None:
                return {
                    "error": "uuid_collision",
                    "id": page.id,
                    "message": "UUID4 collision (extraordinary); retry the call",
                }
            _serialize_and_write_page(target, fm_out, body)
            index_result = _run_indexer(app)

        return {
            "id": page.id,
            "original_id": original_id,
            "path": str(target.relative_to(app.cfg.smalt_dir)),
            "type": page.type.value,
            "mode": "create",
            "mangled": True,
            "index_result": index_result,
        }

    # mode == "update"
    page = PAGE_ADAPTER.validate_python(fm_in)  # re-validate (cheap) to bind .id/.type
    target = _page_target_path(app.cfg.smalt_dir, page)
    with app.mutex.acquire("write_page"):
        if _existing_page_path(app, page.id) is None:
            return {
                "error": "not_found",
                "id": page.id,
                "message": f"page {page.id!r} does not exist; use mode='create' to insert (it will be mangled)",
            }
        _serialize_and_write_page(target, fm_in, body)
        index_result = _run_indexer(app)

    return {
        "id": page.id,
        "path": str(target.relative_to(app.cfg.smalt_dir)),
        "type": page.type.value,
        "mode": "update",
        "mangled": False,
        "index_result": index_result,
    }


# ---- handler: write_pages (READ_WRITE) — batch ----


@_wrap_sync_in_thread
def write_pages(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-write a list of pages with a single indexer pass at the end.

    Same mode semantics as `write_page`: `create` mangles every id;
    `update` requires every id to exist already. The mode applies uniformly
    to the whole batch.

    Validate-all-then-act contract:
      1. Every entry's frontmatter is validated up front.
      2. For `mode='update'`, every entry's id is checked for existence
         before any writes happen.
      3. Any validation or mode-check failure aborts the entire batch;
         the response reports the offending index.

    `mode='create'` cannot fail per-entry on collision (mangling makes
    every id unique by construction), so phase 2's existence check is a
    no-op in that mode.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    items = arguments.get("pages")
    if not items or not isinstance(items, list):
        return {"error": "missing_argument", "message": "pages must be a non-empty list"}
    mode = arguments.get("mode") or "create"
    if mode not in _VALID_WRITE_MODES:
        return {
            "error": "invalid_argument",
            "message": f"mode must be one of {sorted(_VALID_WRITE_MODES)}; got {mode!r}",
        }

    # Phase 1: validate every entry's frontmatter (no disk).
    validated: list[tuple[dict[str, Any], str]] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            return {"error": "validation_error", "index": i, "message": "each entry must be an object"}
        entry = cast("dict[str, Any]", entry)
        fm = entry.get("frontmatter")
        if not fm:
            return {"error": "validation_error", "index": i, "message": "frontmatter is required"}
        body = entry.get("body") or ""
        try:
            PAGE_ADAPTER.validate_python(fm)
        except ValidationError as e:
            return {"error": "validation_error", "index": i, "message": str(e)}
        # IndexPages are auto-generated; reject any in the batch (matches
        # write_page's single-write rejection). Same all-or-nothing
        # contract: the whole batch aborts.
        if fm.get("type") == PageType.INDEX.value:
            return {
                "error": "forbidden_page_type",
                "index": i,
                "type": PageType.INDEX.value,
                "message": (
                    "IndexPages are auto-generated by the indexer; "
                    "direct writes via write_pages are rejected. "
                    f"Entry {i} had type='index'."
                ),
            }
        validated.append((fm, body))

    written: list[dict[str, Any]] = []
    with app.mutex.acquire("write_pages"):
        if mode == "update":
            # Phase 2: every id must already exist (all-or-nothing).
            for i, (fm, _body) in enumerate(validated):
                if _existing_page_path(app, fm["id"]) is None:
                    return {
                        "error": "not_found",
                        "index": i,
                        "id": fm["id"],
                        "message": f"page {fm['id']!r} does not exist; batch aborted",
                    }

        # Phase 3: commit each entry.
        for fm_in, body in validated:
            if mode == "create":
                if "::" in fm_in["id"]:
                    # Section-id upsert path (same as write_page).
                    page = PAGE_ADAPTER.validate_python(fm_in)
                    target = _page_target_path(app.cfg.smalt_dir, page)
                    already_existed = _existing_page_path(app, page.id) is not None
                    _serialize_and_write_page(target, fm_in, body)
                    written.append(
                        {
                            "id": page.id,
                            "path": str(target.relative_to(app.cfg.smalt_dir)),
                            "type": page.type.value,
                            "mangled": False,
                            "upserted": already_existed,
                        }
                    )
                    continue
                page, fm_out, original_id = _prepare_create_write(fm_in)
                target = _page_target_path(app.cfg.smalt_dir, page)
                # UUID collision check (defensive)
                if _existing_page_path(app, page.id) is not None:
                    return {
                        "error": "uuid_collision",
                        "id": page.id,
                        "message": "UUID4 collision (extraordinary); retry the batch",
                    }
                _serialize_and_write_page(target, fm_out, body)
                written.append(
                    {
                        "id": page.id,
                        "original_id": original_id,
                        "path": str(target.relative_to(app.cfg.smalt_dir)),
                        "type": page.type.value,
                        "mangled": True,
                    }
                )
            else:  # update
                page = PAGE_ADAPTER.validate_python(fm_in)
                target = _page_target_path(app.cfg.smalt_dir, page)
                _serialize_and_write_page(target, fm_in, body)
                written.append(
                    {
                        "id": page.id,
                        "path": str(target.relative_to(app.cfg.smalt_dir)),
                        "type": page.type.value,
                        "mangled": False,
                    }
                )

        index_result = _run_indexer(app)

    return {
        "written": written,
        "count": len(written),
        "mode": mode,
        "index_result": index_result,
    }


# ---- handler: add_link (READ_WRITE) ----


def _locate_page_file(app: App, page_id: str) -> Path | None:
    """Return the on-disk path of a page by id, via the LanceDB pages table.

    Returns None if the id isn't indexed. Caller must hold the corpus mutex
    if it intends to mutate the file (the mutex serializes index reads with
    concurrent writes; the LanceDB snapshot we read here may otherwise be
    stale w.r.t. another in-flight write).
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = pages.search().where(f"id = {lance.sql_str(page_id)}").select(["path"]).limit(1).to_arrow()
    if arrow.num_rows == 0:
        return None
    rel = arrow.column("path")[0].as_py()
    return app.cfg.smalt_dir / rel


@_wrap_sync_in_thread
def add_link(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append an outgoing link to a page's `links_out` (read-modify-write).

    Locates the page by id, reads its current frontmatter from disk (not
    LanceDB — we want the latest), appends `{target, label?, via_source?}` to
    `links_out`, writes back atomically, and runs an incremental indexer
    pass. Skips duplicates: a link with the same `target` AND `label`
    already in the list is a no-op and `added: false, reason: duplicate`.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    from_id = arguments.get("from_id")
    to_id = arguments.get("to_id")
    if not from_id:
        return {"error": "missing_argument", "message": "from_id is required"}
    if not to_id:
        return {"error": "missing_argument", "message": "to_id is required"}
    label = arguments.get("label")
    via_source = arguments.get("via_source")

    with app.mutex.acquire("add_link"):
        page_path = _locate_page_file(app, from_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": from_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)

        new_link: dict[str, Any] = {"target": to_id}
        if label is not None:
            new_link["label"] = label
        if via_source is not None:
            new_link["via_source"] = via_source

        existing_links: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("links_out") or [])
        for existing in existing_links:
            if existing.get("target") == to_id and existing.get("label") == label:
                return {
                    "id": from_id,
                    "added": False,
                    "reason": "duplicate",
                    "link": new_link,
                }

        new_fm: dict[str, Any] = dict(parsed.raw_frontmatter)
        new_fm["links_out"] = [*existing_links, new_link]

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": from_id,
        "added": True,
        "link": new_link,
        "links_out_count": len(new_fm["links_out"]),
        "index_result": index_result,
    }


# ---- handler: add_claim (READ_WRITE) ----


@_wrap_sync_in_thread
def add_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append a Claim to a page's `claims` list (read-modify-write).

    Locates the page by id, validates the claim against the `Claim` Pydantic
    model, then reads the page's current frontmatter from disk, appends the
    raw claim dict to `claims`, writes back atomically, and runs the
    indexer. Skips duplicates: a claim with an id already present in the
    list is a no-op and `added: false, reason: duplicate_claim_id`.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    claim = arguments.get("claim")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    if not claim or not isinstance(claim, dict):
        return {"error": "missing_argument", "message": "claim object is required"}

    try:
        validated_claim = Claim.model_validate(claim)
    except ValidationError as e:
        return {"error": "validation_error", "message": str(e)}

    with app.mutex.acquire("add_claim"):
        page_path = _locate_page_file(app, page_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": page_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)

        existing_claims: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("claims") or [])
        for existing in existing_claims:
            if existing.get("id") == validated_claim.id:
                return {
                    "id": page_id,
                    "added": False,
                    "reason": "duplicate_claim_id",
                    "claim_id": validated_claim.id,
                }

        new_fm: dict[str, Any] = dict(parsed.raw_frontmatter)
        # Store the user-provided dict so sparse-on-disk philosophy holds
        # (don't auto-fill optional fields the user omitted).
        new_fm["claims"] = [*existing_claims, claim]

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "added": True,
        "claim_id": validated_claim.id,
        "claims_count": len(new_fm["claims"]),
        "index_result": index_result,
    }


# ---- handler: add_links (READ_WRITE) — batch ----


@_wrap_sync_in_thread
def add_links(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append multiple outgoing links to a page's `links_out` in one batch.

    Validate-all-then-act contract:
      1. Every link in the batch is validated against the `Link` schema.
         Any validation failure aborts the whole batch with
         `{error: 'validation_error', index: N, message: ...}`.
      2. The page is located once (single existence check).
      3. Duplicate detection runs per-item against (a) the page's
         existing `links_out` on disk AND (b) earlier items in this same
         batch. Duplicates are NOT errors — they're reported per-item as
         `{added: false, reason: 'duplicate'}`, matching the single-call
         `add_link` semantics. The non-duplicate items are written
         atomically + the indexer runs once.

    Net: one disk read, one disk write, one indexer pass per call —
    instead of N round-trips for callers that have many links to add
    (M3 ingest's entity-resolution stage is the motivating use case).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    links_in = arguments.get("links")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    if not isinstance(links_in, list) or not links_in:
        return {
            "error": "missing_argument",
            "message": "links must be a non-empty list",
        }

    # Phase 1: validate every entry's structure (no disk).
    from smalt_mcp.schema import Link

    validated_dicts: list[dict[str, Any]] = []
    for i, item in enumerate(links_in):
        if not isinstance(item, dict):
            return {"error": "validation_error", "index": i, "message": "each link must be an object"}
        item = cast("dict[str, Any]", item)
        try:
            Link.model_validate(item)
        except ValidationError as e:
            return {"error": "validation_error", "index": i, "message": str(e)}
        # Preserve the user-supplied dict shape (omit None defaults so the
        # frontmatter stays sparse — mirrors single-call add_link).
        clean: dict[str, Any] = {"target": item["target"]}
        if item.get("label") is not None:
            clean["label"] = item["label"]
        if item.get("via_source") is not None:
            clean["via_source"] = item["via_source"]
        validated_dicts.append(clean)

    results: list[dict[str, Any]] = []
    added_count = 0
    with app.mutex.acquire("add_links"):
        page_path = _locate_page_file(app, page_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": page_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
        existing_links: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("links_out") or [])

        # Build a key-set over (target, label) for fast duplicate detection.
        # Same identity as the single-call add_link uses.
        def _link_key(link: dict[str, Any]) -> tuple[Any, Any]:
            return (link.get("target"), link.get("label"))

        seen_keys: set[tuple[Any, Any]] = {_link_key(existing) for existing in existing_links}
        to_append: list[dict[str, Any]] = []

        for link in validated_dicts:
            key = _link_key(link)
            if key in seen_keys:
                results.append({"added": False, "reason": "duplicate", "link": link})
                continue
            seen_keys.add(key)
            to_append.append(link)
            results.append({"added": True, "link": link})
            added_count += 1

        # If no items actually need adding, skip the write + indexer pass.
        if not to_append:
            return {
                "id": page_id,
                "added_count": 0,
                "duplicate_count": len(results),
                "results": results,
                "links_out_count": len(existing_links),
                "index_result": None,
            }

        new_fm: dict[str, Any] = dict(parsed.raw_frontmatter)
        new_fm["links_out"] = existing_links + to_append
        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "added_count": added_count,
        "duplicate_count": len(results) - added_count,
        "results": results,
        "links_out_count": len(existing_links) + len(to_append),
        "index_result": index_result,
    }


# ---- handler: add_claims (READ_WRITE) — batch ----


@_wrap_sync_in_thread
def add_claims(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append multiple Claims to a page's `claims` list in one batch.

    Same validate-all-then-act contract as `add_links`:
      1. Every claim is validated against the `Claim` schema; any
         validation failure aborts the whole batch.
      2. The page is located once.
      3. Duplicate detection (by claim `id`) runs per-item against the
         existing claims AND against earlier items in this same batch;
         duplicates are reported per-item, not errors.
      4. Non-duplicates are appended atomically + the indexer runs once.

    Net: one disk read, one disk write, one indexer pass per call.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    claims_in = arguments.get("claims")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    if not isinstance(claims_in, list) or not claims_in:
        return {
            "error": "missing_argument",
            "message": "claims must be a non-empty list",
        }

    # Phase 1: validate every entry's structure (no disk).
    validated_pairs: list[tuple[dict[str, Any], str]] = []  # (raw_dict, claim_id)
    for i, item in enumerate(claims_in):
        if not isinstance(item, dict):
            return {"error": "validation_error", "index": i, "message": "each claim must be an object"}
        item = cast("dict[str, Any]", item)
        try:
            validated = Claim.model_validate(item)
        except ValidationError as e:
            return {"error": "validation_error", "index": i, "message": str(e)}
        # Keep the raw user-supplied dict (sparse-on-disk philosophy);
        # use the validated model only for the id (defense against
        # weird casing / extra-fields drift).
        validated_pairs.append((item, validated.id))

    results: list[dict[str, Any]] = []
    added_count = 0
    with app.mutex.acquire("add_claims"):
        page_path = _locate_page_file(app, page_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": page_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
        existing_claims: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("claims") or [])

        existing_ids: set[str] = {cid for c in existing_claims if (cid := c.get("id"))}
        to_append: list[dict[str, Any]] = []

        for raw, claim_id in validated_pairs:
            if claim_id in existing_ids:
                results.append({"added": False, "reason": "duplicate_claim_id", "claim_id": claim_id})
                continue
            existing_ids.add(claim_id)  # block in-batch duplicates from the same id
            to_append.append(raw)
            results.append({"added": True, "claim_id": claim_id})
            added_count += 1

        if not to_append:
            return {
                "id": page_id,
                "added_count": 0,
                "duplicate_count": len(results),
                "results": results,
                "claims_count": len(existing_claims),
                "index_result": None,
            }

        new_fm: dict[str, Any] = dict(parsed.raw_frontmatter)
        new_fm["claims"] = existing_claims + to_append
        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "added_count": added_count,
        "duplicate_count": len(results) - added_count,
        "results": results,
        "claims_count": len(existing_claims) + len(to_append),
        "index_result": index_result,
    }


# ---- handler: write_batch (READ_WRITE) — mixed-op transaction ----

# Op kinds accepted in write_batch. Destructive ops (`remove_page`,
# `remove_link`, `remove_claim`) are NOT accepted in v1 — they're at the
# REMOVE_DESTRUCTIVE tier and including them would force write_batch up a
# tier. Bulk versions (`add_links`, `add_claims`) are also not accepted —
# callers can flatten them into many single-op entries, and supporting
# batch-of-batch raises edge-case complexity for no real win.
_WRITE_BATCH_OP_KINDS: frozenset[str] = frozenset(
    {
        "write_page",
        "add_link",
        "add_claim",
        "update_claim",
    }
)


@_wrap_sync_in_thread
def write_batch(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Mixed-op atomic transaction: pages + links + claims + claim-updates
    in one MCP call. Single indexer pass at the end.

    Each op in the batch is one of:
      - `{kind: "write_page", frontmatter, body?, mode?}` (mode default "create")
      - `{kind: "add_link", from_id, to_id, label?, via_source?}`
      - `{kind: "add_claim", page_id, claim}`
      - `{kind: "update_claim", page_id, claim_id, new_claim}`

    **Three-phase contract** (matches `write_pages`):
      1. Validate-all: every op's `kind` and args are validated before any
         disk work. Any failure aborts the whole batch with
         `{error: 'validation_error', index: N, message}` (or `unknown_kind`
         for unrecognized op kinds).
      2. Existence-check (under corpus mutex): for ops that reference an
         existing page (add_link/add_claim/update_claim), verify the page
         exists in LanceDB. For update_claim, verify the claim id exists
         on that page. Any failure aborts with
         `{error: 'not_found' | 'claim_not_found', index: N}`. NOTE: this
         check uses pre-batch LanceDB state, so an op that targets a page
         CREATED EARLIER IN THE SAME BATCH will be reported as not_found —
         cross-op references inside a single batch are not supported. To
         create a page and then add things to it, use two consecutive
         write_batch calls (or two consecutive tool calls; the auto-indexer
         on the first commit makes the second batch see the new page).
      3. Commit (still under mutex): each op is executed inline (same
         logic as its single-call counterpart but without the per-op
         indexer pass). Disk writes happen sequentially; multiple ops on
         the same page result in multiple read-modify-writes (correct;
         intermediate state is mutex-protected and never visible to
         other readers). The indexer runs **once** at the end.

    Per-op results are returned in input order. On a successful batch:
    `{committed: true, count: N, results: [...], index_result: {...}}`.

    Skipping the indexer per-op is the whole point — for a batch of 20
    mixed ops, the indexer's fastembed call + LanceDB writes happen once
    instead of 20 times.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    ops = arguments.get("ops")
    if not isinstance(ops, list) or not ops:
        return {
            "error": "missing_argument",
            "message": "ops must be a non-empty list",
        }

    # ---- Phase 1: validate every op (no disk) ----

    # Each entry is `(index, kind, validated_args_dict)`. We re-key the
    # args into normalized shapes here so phase 3 doesn't need to re-parse.
    validated: list[tuple[int, str, dict[str, Any]]] = []

    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return {"error": "validation_error", "index": i, "message": "each op must be an object"}
        op = cast("dict[str, Any]", op)
        kind = op.get("kind")
        if kind not in _WRITE_BATCH_OP_KINDS:
            return {
                "error": "unknown_kind",
                "index": i,
                "kind": kind,
                "message": f"kind must be one of {sorted(_WRITE_BATCH_OP_KINDS)}; got {kind!r}",
            }

        if kind == "write_page":
            fm = op.get("frontmatter")
            if not fm:
                return {"error": "validation_error", "index": i, "message": "frontmatter is required"}
            body = op.get("body") or ""
            mode = op.get("mode") or "create"
            if mode not in _VALID_WRITE_MODES:
                return {
                    "error": "invalid_argument",
                    "index": i,
                    "message": f"mode must be one of {sorted(_VALID_WRITE_MODES)}; got {mode!r}",
                }
            try:
                PAGE_ADAPTER.validate_python(fm)
            except ValidationError as e:
                return {"error": "validation_error", "index": i, "message": str(e)}
            if fm.get("type") == PageType.INDEX.value:
                return {
                    "error": "forbidden_page_type",
                    "index": i,
                    "type": PageType.INDEX.value,
                    "message": "IndexPages are auto-generated; direct writes via write_batch are rejected.",
                }
            validated.append((i, kind, {"frontmatter": fm, "body": body, "mode": mode}))

        elif kind == "add_link":
            from_id = op.get("from_id")
            to_id = op.get("to_id")
            if not from_id:
                return {"error": "validation_error", "index": i, "message": "from_id is required"}
            if not to_id:
                return {"error": "validation_error", "index": i, "message": "to_id is required"}
            link_shape: dict[str, Any] = {"target": to_id}
            if op.get("label") is not None:
                link_shape["label"] = op["label"]
            if op.get("via_source") is not None:
                link_shape["via_source"] = op["via_source"]
            try:
                from smalt_mcp.schema import Link

                Link.model_validate(link_shape)
            except ValidationError as e:
                return {"error": "validation_error", "index": i, "message": str(e)}
            validated.append((i, kind, {"from_id": from_id, "link": link_shape}))

        elif kind == "add_claim":
            page_id = op.get("page_id")
            claim = op.get("claim")
            if not page_id:
                return {"error": "validation_error", "index": i, "message": "page_id is required"}
            if not isinstance(claim, dict):
                return {"error": "validation_error", "index": i, "message": "claim object is required"}
            try:
                validated_claim = Claim.model_validate(claim)
            except ValidationError as e:
                return {"error": "validation_error", "index": i, "message": str(e)}
            validated.append(
                (i, kind, {"page_id": page_id, "claim_raw": claim, "claim_id": validated_claim.id})
            )

        else:  # kind == "update_claim"
            page_id = op.get("page_id")
            claim_id = op.get("claim_id")
            new_claim = op.get("new_claim")
            if not page_id:
                return {"error": "validation_error", "index": i, "message": "page_id is required"}
            if not claim_id:
                return {"error": "validation_error", "index": i, "message": "claim_id is required"}
            if not isinstance(new_claim, dict):
                return {"error": "validation_error", "index": i, "message": "new_claim object is required"}
            if new_claim.get("id") != claim_id:
                return {
                    "error": "invalid_argument",
                    "index": i,
                    "message": (
                        f"new_claim.id ({new_claim.get('id')!r}) must match "
                        f"claim_id ({claim_id!r}) — update doesn't rename claims"
                    ),
                }
            try:
                Claim.model_validate(new_claim)
            except ValidationError as e:
                return {"error": "validation_error", "index": i, "message": str(e)}
            validated.append((i, kind, {"page_id": page_id, "claim_id": claim_id, "new_claim": new_claim}))

    # ---- Phase 2 + 3: under corpus mutex ----

    results: list[dict[str, Any]] = []
    with app.mutex.acquire("write_batch"):
        # Phase 2: existence checks for the ops that need them.
        for i, kind, op in validated:
            if kind == "write_page":
                fm = op["frontmatter"]
                mode = op["mode"]
                if mode == "update" and _existing_page_path(app, fm["id"]) is None:
                    return {
                        "error": "not_found",
                        "index": i,
                        "id": fm["id"],
                        "message": f"page {fm['id']!r} does not exist; batch aborted",
                    }
                # mode='create' with slug: no existence check (mangling
                # makes each new id unique by construction).
                # mode='create' with section id: upsert, no existence check.

            elif kind == "add_link":
                if _existing_page_path(app, op["from_id"]) is None:
                    return {
                        "error": "not_found",
                        "index": i,
                        "page_id": op["from_id"],
                        "message": f"page {op['from_id']!r} does not exist; batch aborted",
                    }

            elif kind == "add_claim":
                if _existing_page_path(app, op["page_id"]) is None:
                    return {
                        "error": "not_found",
                        "index": i,
                        "page_id": op["page_id"],
                        "message": f"page {op['page_id']!r} does not exist; batch aborted",
                    }

            else:  # update_claim
                page_path = _locate_page_file(app, op["page_id"])
                if page_path is None or not page_path.exists():
                    return {
                        "error": "not_found",
                        "index": i,
                        "page_id": op["page_id"],
                        "message": f"page {op['page_id']!r} does not exist; batch aborted",
                    }
                # Check claim_id exists on the page — parse the file once
                # (cheap; same file may be touched by later ops, but the
                # claim_id check needs the on-disk state at phase-2 time).
                try:
                    parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
                except (ValidationError, ValueError, OSError) as e:
                    return {
                        "error": "parse_error",
                        "index": i,
                        "page_id": op["page_id"],
                        "message": f"failed to parse {op['page_id']!r}: {e}",
                    }
                existing_claims = parsed.raw_frontmatter.get("claims") or []
                claim_ids = {c.get("id") for c in existing_claims if isinstance(c, dict)}
                if op["claim_id"] not in claim_ids:
                    return {
                        "error": "claim_not_found",
                        "index": i,
                        "page_id": op["page_id"],
                        "claim_id": op["claim_id"],
                        "message": (
                            f"claim {op['claim_id']!r} not found on page {op['page_id']!r}; batch aborted"
                        ),
                    }

        # Phase 3: commit each op in order. Each op writes its own file;
        # the indexer runs ONCE at the end.
        for i, kind, op in validated:
            if kind == "write_page":
                fm_in = op["frontmatter"]
                body = op["body"]
                mode = op["mode"]
                if mode == "create" and "::" in fm_in["id"]:
                    # Section-id upsert path.
                    page = PAGE_ADAPTER.validate_python(fm_in)
                    target = _page_target_path(app.cfg.smalt_dir, page)
                    already_existed = _existing_page_path(app, page.id) is not None
                    _serialize_and_write_page(target, fm_in, body)
                    results.append(
                        {
                            "index": i,
                            "kind": kind,
                            "id": page.id,
                            "path": str(target.relative_to(app.cfg.smalt_dir)),
                            "type": page.type.value,
                            "mode": "create",
                            "mangled": False,
                            "upserted": already_existed,
                        }
                    )
                elif mode == "create":
                    # Slug create — always mangle.
                    page, fm_out, original_id = _prepare_create_write(fm_in)
                    target = _page_target_path(app.cfg.smalt_dir, page)
                    if _existing_page_path(app, page.id) is not None:
                        return {
                            "error": "uuid_collision",
                            "index": i,
                            "id": page.id,
                            "message": "UUID4 collision (extraordinary); retry the batch",
                        }
                    _serialize_and_write_page(target, fm_out, body)
                    results.append(
                        {
                            "index": i,
                            "kind": kind,
                            "id": page.id,
                            "original_id": original_id,
                            "path": str(target.relative_to(app.cfg.smalt_dir)),
                            "type": page.type.value,
                            "mode": "create",
                            "mangled": True,
                        }
                    )
                else:  # mode == "update"
                    page = PAGE_ADAPTER.validate_python(fm_in)
                    target = _page_target_path(app.cfg.smalt_dir, page)
                    _serialize_and_write_page(target, fm_in, body)
                    results.append(
                        {
                            "index": i,
                            "kind": kind,
                            "id": page.id,
                            "path": str(target.relative_to(app.cfg.smalt_dir)),
                            "type": page.type.value,
                            "mode": "update",
                            "mangled": False,
                        }
                    )

            elif kind == "add_link":
                from_id = op["from_id"]
                new_link = op["link"]
                # page_path can't be None — phase 2 verified existence (under
                # the same mutex, so the file can't vanish between phases).
                page_path = cast("Path", _locate_page_file(app, from_id))
                parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
                existing_links = list(parsed.raw_frontmatter.get("links_out") or [])
                duplicate = any(
                    existing.get("target") == new_link["target"]
                    and existing.get("label") == new_link.get("label")
                    for existing in existing_links
                )
                if duplicate:
                    results.append(
                        {
                            "index": i,
                            "kind": kind,
                            "id": from_id,
                            "added": False,
                            "reason": "duplicate",
                            "link": new_link,
                        }
                    )
                    continue
                new_fm = dict(parsed.raw_frontmatter)
                new_fm["links_out"] = [*existing_links, new_link]
                _serialize_and_write_page(page_path, new_fm, parsed.body)
                results.append(
                    {
                        "index": i,
                        "kind": kind,
                        "id": from_id,
                        "added": True,
                        "link": new_link,
                        "links_out_count": len(new_fm["links_out"]),
                    }
                )

            elif kind == "add_claim":
                page_id = op["page_id"]
                claim_raw = op["claim_raw"]
                claim_id = op["claim_id"]
                # page_path can't be None — phase 2 verified existence.
                page_path = cast("Path", _locate_page_file(app, page_id))
                parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
                existing_claims = list(parsed.raw_frontmatter.get("claims") or [])
                duplicate = any(c.get("id") == claim_id for c in existing_claims if isinstance(c, dict))
                if duplicate:
                    results.append(
                        {
                            "index": i,
                            "kind": kind,
                            "id": page_id,
                            "added": False,
                            "reason": "duplicate_claim_id",
                            "claim_id": claim_id,
                        }
                    )
                    continue
                new_fm = dict(parsed.raw_frontmatter)
                new_fm["claims"] = [*existing_claims, claim_raw]
                _serialize_and_write_page(page_path, new_fm, parsed.body)
                results.append(
                    {
                        "index": i,
                        "kind": kind,
                        "id": page_id,
                        "added": True,
                        "claim_id": claim_id,
                        "claims_count": len(new_fm["claims"]),
                    }
                )

            else:  # update_claim
                page_id = op["page_id"]
                claim_id = op["claim_id"]
                new_claim = op["new_claim"]
                # page_path can't be None — phase 2 verified existence.
                page_path = cast("Path", _locate_page_file(app, page_id))
                parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
                claims = list(parsed.raw_frontmatter.get("claims") or [])
                # Phase 2 already verified claim_id presence — but a
                # prior op in this batch might have removed/added claims
                # on the same page. Re-check at commit time too.
                idx = next(
                    (k for k, c in enumerate(claims) if isinstance(c, dict) and c.get("id") == claim_id),
                    None,
                )
                if idx is None:
                    return {
                        "error": "claim_not_found",
                        "index": i,
                        "page_id": page_id,
                        "claim_id": claim_id,
                        "message": (
                            f"claim {claim_id!r} not found on {page_id!r} at commit time "
                            "(earlier op in this batch removed it?)"
                        ),
                    }
                claims[idx] = new_claim
                new_fm = dict(parsed.raw_frontmatter)
                new_fm["claims"] = claims
                _serialize_and_write_page(page_path, new_fm, parsed.body)
                results.append(
                    {
                        "index": i,
                        "kind": kind,
                        "id": page_id,
                        "claim_id": claim_id,
                        "updated": True,
                    }
                )

        # One indexer pass at the end.
        index_result = _run_indexer(app)

    return {
        "committed": True,
        "count": len(results),
        "results": results,
        "index_result": index_result,
    }


# ---- handler: reindex_page + reindex_all (READ_WRITE) — granular ops ----

# C-9: explicit re-index entry points. The auto-trigger on every write
# covers the common case (just-edited page → indexer runs → table is
# consistent), but two real workflows want explicit re-indexing:
#
# 1. **Post-restore from a Restic backup that excluded `index/lance/`**
#    (the pattern documented in README's "Operations: backup and
#    restore"). The markdown is on disk; LanceDB is empty or stale.
#    `reindex_all` rebuilds from `pages/`. This is the explicit
#    counterpart to `bootstrap` (which conflates first-init with
#    re-init).
#
# 2. **M7 Cogitate / M6 Research flows** where a single page's content
#    changed via a non-write_page path — e.g., a hand-edit on disk, or a
#    proposal-apply that updated a summary. `reindex_page(page_id)`
#    forces re-projection without sweeping the whole corpus.


def _find_page_file_by_id(smalt_root: Path, page_id: str) -> Path | None:
    """Walk `pages/` and find the markdown file whose frontmatter `id`
    matches `page_id`. Returns None if no file matches.

    Used by `reindex_page` for the case where the page isn't in
    LanceDB yet (e.g., restored from disk; never indexed). Cheap fall
    back — we already have LanceDB as the fast path; this is only
    invoked when LanceDB doesn't know about the id.
    """
    pages_root = paths.pages_dir(smalt_root)
    if not pages_root.exists():
        return None
    for path in pages_root.rglob("*.md"):
        if not path.is_file():
            continue
        try:
            parsed = parse_page(path, smalt_root=smalt_root)
        except (ValidationError, ValueError, OSError):
            continue
        if parsed.frontmatter.id == page_id:
            return path
    return None


@_wrap_sync_in_thread
def reindex_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Force-re-index a single page from disk.

    Locates the page (LanceDB lookup first, filesystem walk as
    fallback), re-parses it, projects to `pages` + `embeddings` +
    `links` + `claims` tables, refreshes FTS + ANN. Returns the
    per-page indexer summary.

    Use cases:
      - The page was edited on disk outside the MCP write path (rare
        but real — humans poking at the markdown directly).
      - A proposal-apply via cobalt-grinding's orchestration updated
        the page's body and the agent wants re-embedding even though
        write_page already triggered the indexer (defensive belt-and-
        suspenders for high-stakes edits).
      - The page is on disk (e.g., post-restore) but not yet indexed.

    Runs under the corpus mutex; one indexer pass for the single file.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}

    with app.mutex.acquire("reindex_page"):
        # Try LanceDB first; fall back to filesystem walk if the page
        # isn't indexed yet.
        rel_path = _existing_page_path(app, page_id)
        if rel_path is not None:
            file_path = app.cfg.smalt_dir / rel_path
            if not file_path.exists():
                # Indexed row points at a missing file — the caller
                # probably wants remove_page, not reindex_page. Surface
                # both so they can choose.
                return {
                    "error": "file_not_found",
                    "page_id": page_id,
                    "indexed_path": rel_path,
                    "message": (
                        f"page {page_id!r} is indexed at {rel_path!r} but the file is gone; "
                        "use remove_page to drop the orphaned row, or restore the file first"
                    ),
                }
        else:
            file_path = _find_page_file_by_id(app.cfg.smalt_dir, page_id)
            if file_path is None:
                return {
                    "error": "not_found",
                    "page_id": page_id,
                    "message": (
                        f"page {page_id!r} not found in LanceDB or on disk; "
                        "check the id (try find_by_alias if you have a memorable handle)"
                    ),
                }

        from smalt_mcp.storage.indexer import Indexer

        indexer = Indexer(
            smalt_root=app.cfg.smalt_dir,
            embedder=app.embedder(),
            db=app.db(),
        )
        result = indexer.run(only_paths=[file_path])
        app.record_indexer_run(result)

    return {
        "page_id": page_id,
        "path": str(file_path.relative_to(app.cfg.smalt_dir)),
        "index_result": result.to_dict(),
    }


async def reindex_all(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Full LanceDB rebuild from disk (C-13: now async).

    Enqueues a background task that wipes every table, recreates the
    schemas, walks `pages/`, and projects everything. Returns
    immediately with a `task_id`; clients poll via `task_status` to
    track state and read the final IndexResult when the task reaches
    `succeeded`.

    The thorough version of `Indexer.run(full=True)` — handles the case
    where the LanceDB tables themselves are missing or corrupt (e.g.,
    post-restore from a Restic backup that excluded `index/lance/`).

    The task runs under the corpus mutex for the whole rebuild; on a
    multi-GB Smalt this can take minutes. Pre-C-13 callers that
    expected a synchronous result + `wiped_tables` / `recreated_tables`
    / `index_result` payload should now poll `task_status(task_id)`
    until state is terminal; the same payload appears in `task.result`.

    Response shape:

        {
          "task_id": str,
          "kind": "reindex_all",
          "state": "pending",
          "created_at": ISO datetime,
          "message": "Enqueued; poll task_status(task_id) for progress."
        }
    """
    if not app.smalt_exists():
        return _not_initialized()

    async def _work(task: Task) -> dict[str, Any]:
        """The actual reindex work. Runs under the corpus mutex; uses
        `asyncio.to_thread` so the blocking LanceDB ops don't block the
        event loop."""

        def _do_reindex() -> dict[str, Any]:
            task.progress["phase"] = "acquiring_mutex"
            with app.mutex.acquire("reindex_all"):
                # Honor cancel before any destructive work happens.
                if task.cancel_requested:
                    raise asyncio.CancelledError(f"task {task.id} cancelled before reindex started")

                task.progress["phase"] = "wiping_tables"
                db = app.db()
                wiped: list[str] = []
                for table_name in lance.ALL_TABLES:
                    try:
                        db.drop_table(table_name)
                        wiped.append(table_name)
                    except (FileNotFoundError, ValueError, KeyError):
                        continue
                    except Exception as e:
                        logger.warning(
                            "reindex_all: drop_table(%s) failed: %s",
                            table_name,
                            e,
                        )
                task.progress["wiped_tables"] = list(wiped)

                if task.cancel_requested:
                    raise asyncio.CancelledError(f"task {task.id} cancelled after wipe")

                task.progress["phase"] = "recreating_tables"
                recreated = lance.ensure_tables(app.cfg.smalt_dir, embedding_dim=app.cfg.embedding.dim)
                task.progress["recreated_tables"] = list(recreated)

                task.progress["phase"] = "indexing"
                from smalt_mcp.storage.indexer import Indexer

                indexer = Indexer(
                    smalt_root=app.cfg.smalt_dir,
                    embedder=app.embedder(),
                    db=app.db(),
                )
                result = indexer.run(full=True)
                app.record_indexer_run(result)

                task.progress["phase"] = "complete"
                return {
                    "wiped_tables": wiped,
                    "recreated_tables": recreated,
                    "index_result": result.to_dict(),
                }

        return await asyncio.to_thread(_do_reindex)

    task = app.scheduler.enqueue("reindex_all", _work)
    payload = task.to_dict()
    payload["message"] = (
        "Enqueued; poll task_status(task_id) for progress. "
        "Final result available in task.result when state is 'succeeded'."
    )
    return payload


# ---- handler: task_cancel (READ_WRITE, C-13) ----


@_wrap_sync_in_thread
def task_cancel(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Request cancellation of a scheduled task by `task_id`.

    Cancellation is **cooperative** — the work function must check
    `task.check_cancel()` at safe boundaries to honor the request.
    For `reindex_all`, the worst-case latency between cancel-request
    and actual stop is "one indexer iteration" (typically <1s).

    State transitions:
      - PENDING task → straight to CANCELLED; never runs.
      - RUNNING task → `cancel_requested` flag set; the work function
        bails at its next safe boundary. State stays RUNNING until
        that happens (the response reflects the moment of cancel,
        not the eventual transition — poll task_status to see the
        final CANCELLED state).
      - terminal task → no-op; returns the existing task unchanged.

    Returns the task's current state (post-cancel-request). Errors:
    `not_found` if `task_id` is unknown.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    task_id = arguments.get("task_id")
    if not task_id:
        return {"error": "missing_argument", "message": "task_id is required"}
    task = app.scheduler.cancel(task_id)
    if task is None:
        return {
            "error": "not_found",
            "task_id": task_id,
            "message": (f"no task with id {task_id!r} — either it never existed or it was GC'd"),
        }
    payload = task.to_dict()
    payload["was_terminal_at_call"] = task.state in TERMINAL_STATES and (
        not task.cancel_requested or task.state == TaskState.CANCELLED
    )
    return payload


# ---- handler: remove_page (REMOVE_DESTRUCTIVE) ----


@_wrap_sync_in_thread
def remove_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Cascading delete of a page by canonical id.

    Removes, all under the corpus mutex:
      - the `.md` file from disk (`pages/<subdir>/<id>.md`)
      - the `pages` row
      - the `embeddings` row (page_id match)
      - every outgoing link (from_id match)  ← the page is "leaving"
      - every incoming link (to_id match)    ← references to the gone page
      - every claim attached to the page (page_id match)

    Use `incoming_links(page_id)` first if you want to audit what
    references will be silently dropped.

    No alias resolution: caller must pass the canonical id (use `find_by_alias`
    or `read_page` to resolve from an alias first).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}

    with app.mutex.acquire("remove_page"):
        # Locate the page on disk (also serves as exists-check)
        rel_path = _existing_page_path(app, page_id)
        if rel_path is None:
            return {"error": "not_found", "page_id": page_id}
        abs_path = app.cfg.smalt_dir / rel_path

        # Cascade: delete from each LanceDB table, then the file.
        db = app.db()
        quoted = lance.sql_str(page_id)

        # Count what we're about to remove, for the response.
        links_table = db.open_table(lance.TABLE_LINKS)
        outgoing_n = links_table.search().where(f"from_id = {quoted}").to_arrow().num_rows
        incoming_n = links_table.search().where(f"to_id = {quoted}").to_arrow().num_rows
        claims_table = db.open_table(lance.TABLE_CLAIMS)
        claims_n = claims_table.search().where(f"page_id = {quoted}").to_arrow().num_rows

        # Delete rows
        db.open_table(lance.TABLE_PAGES).delete(f"id = {quoted}")
        db.open_table(lance.TABLE_EMBEDDINGS).delete(f"page_id = {quoted}")
        links_table.delete(f"from_id = {quoted}")
        links_table.delete(f"to_id = {quoted}")
        claims_table.delete(f"page_id = {quoted}")

        # Delete the file (after the index is clear, so a crash mid-op leaves
        # the file present and the index missing — the indexer will surface a
        # not-yet-indexed page rather than a phantom row pointing at nothing).
        if abs_path.exists():
            abs_path.unlink()

    return {
        "id": page_id,
        "removed": {
            "file": str(rel_path),
            "outgoing_links": outgoing_n,
            "incoming_links": incoming_n,
            "claims": claims_n,
            "embedding": 1,  # always 1 if the page was indexed
        },
    }


# ---- handler: update_claim (REMOVE_DESTRUCTIVE) ----


@_wrap_sync_in_thread
def update_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace one claim on a page, identified by `claim_id` within the
    page's `claims` list. Read-modify-write under the corpus mutex.

    The new claim is validated against the `Claim` schema. The claim id
    in `new_claim` must match `claim_id` (we don't let updates change the
    identifier — use add_claim + remove_claim if you want to rename).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    claim_id = arguments.get("claim_id")
    new_claim = arguments.get("new_claim")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    if not claim_id:
        return {"error": "missing_argument", "message": "claim_id is required"}
    if not new_claim or not isinstance(new_claim, dict):
        return {"error": "missing_argument", "message": "new_claim object is required"}
    if new_claim.get("id") != claim_id:
        return {
            "error": "invalid_argument",
            "message": f"new_claim.id ({new_claim.get('id')!r}) must match claim_id ({claim_id!r})",
        }
    try:
        Claim.model_validate(new_claim)
    except ValidationError as e:
        return {"error": "validation_error", "message": str(e)}

    with app.mutex.acquire("update_claim"):
        page_path = _locate_page_file(app, page_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": page_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
        claims: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("claims") or [])
        idx = next((i for i, c in enumerate(claims) if c.get("id") == claim_id), None)
        if idx is None:
            return {
                "error": "claim_not_found",
                "page_id": page_id,
                "claim_id": claim_id,
            }
        claims[idx] = new_claim
        new_fm = dict(parsed.raw_frontmatter)
        new_fm["claims"] = claims

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "claim_id": claim_id,
        "updated": True,
        "index_result": index_result,
    }


# ---- handler: remove_claim (REMOVE_DESTRUCTIVE) ----


@_wrap_sync_in_thread
def remove_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Remove one claim from a page by `claim_id`. RMW under the mutex.

    Returns `{error: 'claim_not_found'}` if the claim id isn't on the page.
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    page_id = arguments.get("page_id")
    claim_id = arguments.get("claim_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}
    if not claim_id:
        return {"error": "missing_argument", "message": "claim_id is required"}

    with app.mutex.acquire("remove_claim"):
        page_path = _locate_page_file(app, page_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": page_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
        claims: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("claims") or [])
        new_claims = [c for c in claims if c.get("id") != claim_id]
        if len(new_claims) == len(claims):
            return {
                "error": "claim_not_found",
                "page_id": page_id,
                "claim_id": claim_id,
            }
        new_fm = dict(parsed.raw_frontmatter)
        new_fm["claims"] = new_claims

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "claim_id": claim_id,
        "removed": True,
        "claims_remaining": len(new_claims),
        "index_result": index_result,
    }


# ---- handler: remove_link (REMOVE_DESTRUCTIVE) ----


@_wrap_sync_in_thread
def remove_link(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Remove an outgoing link from a page, matched by (target, label).

    If `label` is omitted, removes EVERY edge from `from_id` to `to_id`
    regardless of label (returns a count). Otherwise removes only edges
    with matching `(target, label)`.

    RMW under the mutex. Returns `{removed: <count>}`; 0 means no matching
    link existed (not an error — symmetric with add_link's duplicate-no-op).
    """
    err = _ensure_initialized(app)
    if err is not None:
        return err

    from_id = arguments.get("from_id")
    to_id = arguments.get("to_id")
    if not from_id:
        return {"error": "missing_argument", "message": "from_id is required"}
    if not to_id:
        return {"error": "missing_argument", "message": "to_id is required"}
    label = arguments.get("label")  # may be None → match-any-label

    with app.mutex.acquire("remove_link"):
        page_path = _locate_page_file(app, from_id)
        if page_path is None or not page_path.exists():
            return {"error": "not_found", "page_id": from_id}

        parsed = parse_page(page_path, smalt_root=app.cfg.smalt_dir)
        links: list[dict[str, Any]] = list(parsed.raw_frontmatter.get("links_out") or [])

        def matches(link: dict[str, Any]) -> bool:
            if link.get("target") != to_id:
                return False
            if label is None:
                return True
            return link.get("label") == label

        kept = [link for link in links if not matches(link)]
        removed = len(links) - len(kept)
        if removed == 0:
            return {
                "id": from_id,
                "removed": 0,
                "reason": "no_matching_link",
            }
        new_fm = dict(parsed.raw_frontmatter)
        new_fm["links_out"] = kept

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": from_id,
        "removed": removed,
        "links_remaining": len(kept),
        "index_result": index_result,
    }


# ---- registry ----


TOOLS: list[ToolDef] = [
    ToolDef(
        spec=types.Tool(
            name="status",
            description=(
                "Report the current state of the Smalt this server is wrapping: "
                "configured path, whether the directory exists, which LanceDB "
                "tables are present, page count, single-writer mutex state, and "
                "configured embedding provider. Always safe to call; no side "
                "effects. Useful as a first call to verify the server is wired "
                "up correctly and pointed at the expected Smalt."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=status,
    ),
    ToolDef(
        spec=types.Tool(
            name="index_status",
            description=(
                "Report detailed index + indexer state — the deeper "
                "cousin of `status`. Returns:\n"
                "  - Smalt path + existence\n"
                "  - Per-table row counts (pages, embeddings, links, "
                "claims, sources)\n"
                "  - Last indexer run metadata (timestamp, duration, "
                "full IndexResult — inserted/updated/deleted/failed "
                "counts + per-file failures)\n"
                "  - Per-index build status:\n"
                "    - **FTS**: per-field `{status: ok|failed, error: …}` "
                "for `title` + `body`. A `failed` here means the rebuild "
                "raised on the most recent indexer pass and that field's "
                "FTS index may be stale or missing.\n"
                "    - **ANN (vector)**: `{status: ok|failed|skipped, "
                "reason: …}`. `skipped` is the normal state for small "
                "Smalts (<256 embeddings → brute-force scan, no index "
                "needed); `failed` flags a real problem.\n"
                "  - Embedding config + whether the fastembed model is "
                "loaded into memory\n"
                "  - Mutex contention (locked-now, holder, "
                "acquire_count, total_wait_seconds, mean_wait_ms)\n\n"
                "Always safe to call (no writes, no LanceDB mutations); "
                "returns nulls for not-yet-known fields when called "
                "pre-bootstrap or before any indexer pass.\n\n"
                "Mirrors the `GET /admin/health` HTTP route's payload "
                "exactly — use whichever channel is more convenient."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=index_status,
    ),
    ToolDef(
        spec=types.Tool(
            name="list_pages",
            description=(
                "List indexed pages in the Smalt with optional filters. "
                "Returns minimal metadata per page (id, title, type, path) "
                "— use `read_page` to fetch a single page's full body + "
                "frontmatter.\n\n"
                "**LanceDB-side filters** (fast):\n"
                "- `type` — entity / concept / source / synthesis\n"
                "- `prefix` — id starts with (e.g. 'ent-' or 'con-embedding')\n\n"
                "**Property filters** (client-side post-fetch; slower for "
                "huge Smalts but expressive — same semantics shared with "
                "`search`):\n"
                "- `glossary: bool` — ConceptPage `glossary` flag\n"
                "- `is_domain: bool` — ConceptPage `is_domain` flag\n"
                "- `domain: str` — page's `domains:` list contains the given id\n"
                "- `fetched_at_before` / `fetched_at_after`: ISO 8601 datetime — "
                "SourcePage `fetched_at` range\n"
                "- `has_aliases_containing: str` — page's `aliases:` list contains the given value\n\n"
                "All filters AND-composed. Bool filters compare the raw "
                "frontmatter value (a ConceptPage that omits `glossary:` "
                "won't match `glossary=False` — schema defaults don't "
                "materialize at filter time). `limit` applies to the final "
                "filtered set; if the over-fetch budget is exhausted before "
                "`limit` filtered hits accumulate, the response sets "
                "`truncated: true`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Optional page-type filter.",
                        "enum": ["entity", "concept", "source", "synthesis", "index"],
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Optional id-prefix filter (e.g. 'ent-' or 'con-embedding').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max pages to return. Default 100.",
                        "default": 100,
                    },
                    "glossary": {
                        "type": "boolean",
                        "description": "Filter on ConceptPage `glossary` flag.",
                    },
                    "is_domain": {
                        "type": "boolean",
                        "description": "Filter on ConceptPage `is_domain` flag.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter: page's `domains:` list contains this id.",
                    },
                    "fetched_at_before": {
                        "type": "string",
                        "description": (
                            "Filter: SourcePage `fetched_at` strictly before "
                            "this ISO 8601 datetime. Non-source pages don't "
                            "match (SQL-NULL semantics)."
                        ),
                    },
                    "fetched_at_after": {
                        "type": "string",
                        "description": (
                            "Filter: SourcePage `fetched_at` strictly after "
                            "this ISO 8601 datetime. Non-source pages don't "
                            "match (SQL-NULL semantics)."
                        ),
                    },
                    "has_aliases_containing": {
                        "type": "string",
                        "description": "Filter: page's `aliases:` list contains this value.",
                    },
                },
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=list_pages,
    ),
    ToolDef(
        spec=types.Tool(
            name="read_page",
            description=(
                "Return one page's full body + parsed frontmatter.\n\n"
                "Lookup order:\n"
                "  1. Exact match on the canonical `id`.\n"
                "  2. Exact alias fallback: if no exact match, search "
                "every page's `aliases` for `page_id`.\n"
                "      - 1 match → return that page; response includes "
                "`resolved_via_alias: <the alias>`.\n"
                "      - 2+ matches → `{error: 'ambiguous_alias', "
                "matches: [...]}` (caller must pick a canonical id).\n"
                "      - 0 matches → fall through to step 3.\n"
                "  3. **Fuzzy alias fallback** (C-11; default on, opt-out "
                "via `fuzzy: false`): trigram-Jaccard match `page_id` "
                "against every page's aliases.\n"
                "      - 1 match → return that page; response includes "
                "`resolved_via_alias` + `fuzzy: true` + `fuzzy_score` + "
                "`matched_alias`.\n"
                "      - 2+ matches → `{error: 'ambiguous_alias', "
                "matches: [...], fuzzy: true}`.\n"
                "      - 0 matches → `{error: 'not_found', "
                "fuzzy: true}` (the flag confirms the fuzzy pass ran).\n\n"
                "Threshold for the fuzzy step is configurable via the "
                "`SMALT_FUZZY_ALIAS_THRESHOLD` env var (default 0.6). "
                "Use `find_by_alias` if you want the full match list "
                "without picking one."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": (
                            "Canonical page id (e.g. 'ent-alice-XYZ…'), "
                            "a known alias (e.g. 'ent-alice'), or a "
                            "close-but-misspelled handle (e.g. "
                            "'ent-alic'). Exact-id → exact-alias → "
                            "fuzzy-alias resolution runs automatically."
                        ),
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": (
                            "Opt out of the fuzzy alias fallback. "
                            "Default true. Set false for callers that "
                            "need exact-only resolution (e.g. "
                            "automation that distinguishes typo from "
                            "true-not-found)."
                        ),
                        "default": True,
                    },
                },
                "required": ["page_id"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=read_page,
    ),
    ToolDef(
        spec=types.Tool(
            name="find_by_alias",
            description=(
                "List every page whose `aliases` contains `alias`. "
                "Returns minimal metadata per match (id, title, type, "
                "path); call `read_page` on a canonical id to get the "
                "body. Useful when you have a memorable handle (the "
                "original caller-id before write_page mangling, or any "
                "hand-added alias) and want to see which page(s) it maps "
                "to.\n\n"
                "Resolution (C-11):\n"
                "  1. Exact alias match. If any → return with "
                "`fuzzy: false`.\n"
                "  2. If `fuzzy: true` (default) and step 1 found "
                "nothing → trigram-Jaccard fuzzy match. Returned rows "
                "carry `fuzzy_score` (∈ [threshold, 1.0]) and "
                "`matched_alias` (the alias that scored highest for that "
                "page). Top-level `fuzzy: true` plus `fuzzy_threshold` "
                "report the fallback fired and at what bar.\n"
                "  3. If still none → `count: 0` (no error).\n\n"
                "Threshold via `SMALT_FUZZY_ALIAS_THRESHOLD` env var "
                "(default 0.6)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "The alias (or near-alias) to look up.",
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": ("Opt out of the fuzzy fallback (exact-only match). Default true."),
                        "default": True,
                    },
                },
                "required": ["alias"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=find_by_alias,
    ),
    ToolDef(
        spec=types.Tool(
            name="traverse",
            description=(
                "Multi-hop outgoing-link graph traversal via BFS.\n\n"
                "Returns every edge discovered within `hops` hops of "
                "`from_id`, plus `visited_nodes` — the set of nodes "
                "touched (including `from_id` as the seed). Optional "
                "`label` filter is applied **per hop** (only edges with "
                "that label are followed; nodes reachable only via other "
                "labels don't expand).\n\n"
                "Cycle handling: revisited nodes don't re-expand. "
                "Self-loops are collected but don't re-expand. `hops` "
                "defaults to 1; max is 5 (calls above the ceiling return "
                "`invalid_argument`). Per-hop edge query is capped at "
                "1000 rows; if a hop hits the cap, the response sets "
                "`truncated: true` so the caller knows the walk was "
                "incomplete."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "The starting page id.",
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Optional edge-label filter, applied per hop (only matching edges are followed)."
                        ),
                    },
                    "hops": {
                        "type": "integer",
                        "description": (
                            "Number of hops to walk via BFS. Default 1; "
                            "max 5. Values outside [1, 5] return "
                            "`invalid_argument`."
                        ),
                        "default": 1,
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["from_id"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=traverse,
    ),
    ToolDef(
        spec=types.Tool(
            name="incoming_links",
            description=(
                "List every link whose `to_id` matches this page id — the "
                "'what points at me' view. Symmetric to `traverse` (which "
                "lists outgoing links). Each returned edge is "
                "`{from_id, to_id, label, source_page}`. Use this before "
                "`remove_page` to audit what references will be silently "
                "dropped when the page is deleted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The page id being referenced (= the link's `to_id`).",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional edge-label filter.",
                    },
                },
                "required": ["page_id"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=incoming_links,
    ),
    ToolDef(
        spec=types.Tool(
            name="search",
            description=(
                "Hybrid search over the Smalt's pages, three retrieval "
                "sources fused via Reciprocal Rank Fusion:\n"
                "  1. FTS over title + body (`pages` table indexes).\n"
                "  2. Vector similarity over summary embeddings "
                "(`embeddings` table).\n"
                "  3. Alias match: any page whose `aliases` list contains "
                "the query verbatim — or any whitespace-separated token "
                "of the query verbatim — joins the ranking. Catches "
                "searches for caller-ids of mangled pages "
                "(e.g. `search('ent-alice')` finds a page whose canonical "
                "id is `ent-alice-XYZ` because `ent-alice` is in its "
                "`aliases`).\n\n"
                "**Optional property filters** (same semantics as "
                "`list_pages`; applied to hydrated candidates AFTER RRF "
                "fusion and BEFORE top_k truncation, so the top_k you get "
                "back is 'top_k filter-passing matches' — not 'top_k "
                "matches of which some happen to pass'):\n"
                "  - `glossary: bool`, `is_domain: bool`\n"
                "  - `domain: str` (page's `domains:` list contains the id)\n"
                "  - `fetched_at_before` / `fetched_at_after` (ISO 8601)\n"
                "  - `has_aliases_containing: str`\n\n"
                "Returns top-`top_k` matches; each hit carries `id` "
                "(canonical), `aliases`, `title`, `type`, `snippet`, and "
                "`score` (RRF). Empty hit list if no retrieval source "
                "matched or no candidate passed the filters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return. Default 10.",
                        "default": 10,
                    },
                    "glossary": {
                        "type": "boolean",
                        "description": "Filter on ConceptPage `glossary` flag.",
                    },
                    "is_domain": {
                        "type": "boolean",
                        "description": "Filter on ConceptPage `is_domain` flag.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter: page's `domains:` list contains this id.",
                    },
                    "fetched_at_before": {
                        "type": "string",
                        "description": (
                            "Filter: SourcePage `fetched_at` strictly before this "
                            "ISO 8601 datetime. Non-source pages don't match."
                        ),
                    },
                    "fetched_at_after": {
                        "type": "string",
                        "description": (
                            "Filter: SourcePage `fetched_at` strictly after this "
                            "ISO 8601 datetime. Non-source pages don't match."
                        ),
                    },
                    "has_aliases_containing": {
                        "type": "string",
                        "description": "Filter: page's `aliases:` list contains this value.",
                    },
                },
                "required": ["query"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=search,
    ),
    ToolDef(
        spec=types.Tool(
            name="list_domains",
            description=(
                "List ConceptPages flagged `is_domain: true` — the Smalt's "
                "first-class domains. Domain hierarchy itself (which domain "
                "is a subdomain of which) is not in this response; use "
                "`traverse(from_id=<domain>, label='subdomain_of')` to walk "
                "the hierarchy."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=list_domains,
    ),
    ToolDef(
        spec=types.Tool(
            name="source_similarity",
            description=(
                "Pages most similar to `source_id`'s embedding by cosine "
                "similarity. Uses the page's stored vector as the query "
                "— no re-embed needed. Excludes the source itself.\n\n"
                "Returns up to `top_k` matches (default 10), each with "
                "`id`, `title`, `type`, `path`, `aliases`, and "
                "`similarity` (= 1 - cosine_distance; range nominally "
                "[-1, 1], practically [0, 1] for normalized "
                "embeddings; 1.0 = identical).\n\n"
                "Use cases: Cogitate-style discovery ('what other pages "
                "are about something similar to this one?'), de-dup "
                "candidate search across SourcePages (filter "
                "`types: ['source']`), neighborhood scoping for "
                "observer walks that don't follow explicit links.\n\n"
                "**`source_id` is conventionally named** — any indexed "
                "page works as the query vector source, not just "
                "SourcePages. The `types` filter restricts the result "
                "set, not the input.\n\n"
                "Errors: `missing_argument` (source_id omitted); "
                "`invalid_argument` (top_k≤0, types malformed); "
                "`not_found` (source_id has no embedding — likely not "
                "indexed; try `reindex_page` first); "
                "`vector_search_failed` (LanceDB raised — usually "
                "means the ANN index is in a bad state, check "
                "`index_status`)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": (
                            "Canonical page id whose embedding becomes the query vector. Any page type works."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of similar pages to return "
                            "(default 10). Source is excluded; the "
                            "count is over non-source matches."
                        ),
                        "default": 10,
                        "minimum": 1,
                    },
                    "types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "entity",
                                "concept",
                                "source",
                                "synthesis",
                                "index",
                            ],
                        },
                        "description": (
                            "Optional page-type filter — only "
                            "results whose `type` is in this list "
                            "are returned. Applied client-side after "
                            "vector retrieval (5x over-fetch keeps "
                            "the post-filter pool deep). When set, "
                            "the response carries "
                            "`truncated: true` if the filter "
                            "exhausted the over-fetched pool before "
                            "`top_k` matches accumulated."
                        ),
                    },
                },
                "required": ["source_id"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=source_similarity,
    ),
    ToolDef(
        spec=types.Tool(
            name="task_status",
            description=(
                "Return the current state of one scheduled task "
                "(C-13 async task model). Tasks transition pending "
                "→ running → terminal (succeeded / failed / "
                "cancelled). Once terminal, they stay in the "
                "registry for ~1 hour (GC TTL) so clients can poll "
                "the result later.\n\n"
                "Response: full task dict — `task_id`, `kind`, "
                "`state`, timestamps, `progress` (kind-specific "
                "dict), `result` (work-function return value, "
                "populated only on `succeeded`), `error` (short "
                "string, populated only on `failed`), "
                "`cancel_requested` bool.\n\n"
                "Errors: `not_found` if `task_id` is unknown (id "
                "never existed or was GC'd)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "The id returned by the heavy-op tool "
                            "that enqueued the task "
                            "(e.g. `reindex_all`)."
                        ),
                    },
                },
                "required": ["task_id"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=task_status,
    ),
    ToolDef(
        spec=types.Tool(
            name="task_list",
            description=(
                "List scheduled tasks, most-recent first, optionally "
                "filtered by `state` and/or `kind` (C-13 async task "
                "model). `limit` is applied after filtering.\n\n"
                "Returns `{tasks: [...], count, state_filter, "
                "kind_filter}` where each task entry is the same "
                "shape as `task_status`. Useful for ops introspection "
                '("what\'s currently running?" → '
                "`task_list(state='running')`) or debugging recent "
                'failures ("why did my last reindex fail?" → '
                "`task_list(state='failed', kind='reindex_all', "
                "limit=5)`)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": [
                            "pending",
                            "running",
                            "succeeded",
                            "failed",
                            "cancelled",
                        ],
                        "description": (
                            "Optional state filter. When set, only tasks in that state are returned."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "description": (
                            "Optional operation-kind filter "
                            "(e.g. `'reindex_all'`). When set, only "
                            "tasks of that kind are returned."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": ("Max number of tasks to return (default 100)."),
                        "default": 100,
                        "minimum": 1,
                    },
                },
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=task_list,
    ),
    # ---- READ_WRITE ----
    ToolDef(
        spec=types.Tool(
            name="bootstrap",
            description=(
                "Initialize an empty Smalt at the configured SMALT_DIR. "
                "Creates the canonical directory layout, drops in SCHEMA.md / "
                "POLICY.md / tasks/gaps.md placeholders, and creates the "
                "LanceDB tables. Idempotent — running it on an existing "
                "Smalt is a no-op; the response reports only what was newly "
                "created."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=bootstrap,
    ),
    ToolDef(
        spec=types.Tool(
            name="write_page",
            description=(
                "Write one page (frontmatter + body) and trigger an "
                "incremental indexer pass.\n\n"
                "Two modes:\n"
                "  - `create` (DEFAULT): always produces a NEW page. The "
                "caller's id becomes the slug-prefix; a 22-char URL-safe "
                "base64 UUID4 suffix is appended to make the canonical id "
                "structurally unique (collision is impossible). The "
                "original id is preserved in the page's `aliases` list. "
                "Response includes `id` (canonical), `original_id` (what "
                "the caller sent), and `mangled: true`. Callers must store "
                "the canonical id to address the page later.\n"
                "  - `update`: the caller's id must be an existing canonical "
                "id (no mangling, no alias resolution). Fails "
                "`{error: 'not_found'}` if no such page is indexed. Use "
                "this to modify a specific known page.\n\n"
                "Path: `pages/<subdir>/<canonical-id>.md`. Atomic "
                "tmp-then-rename. Write + indexer run inside the corpus "
                "single-writer mutex; the existence check (for update) is "
                "in the same critical section."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "frontmatter": {
                        "type": "object",
                        "description": (
                            "Full page frontmatter. Required keys: id, "
                            "type, title. Other keys per the page-type "
                            "schema (see smalt_mcp/schema.py). The id is "
                            "validated for path-safety + portability "
                            "(alphanumeric + underscore + hyphen, no "
                            "leading dash/underscore, 1..254 chars, not a "
                            "Windows-reserved name)."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": "Page body (everything after the frontmatter block).",
                        "default": "",
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "Write mode. `create` (default) always creates "
                            "a new page with a mangled canonical id. "
                            "`update` requires the caller's id to be an "
                            "existing canonical id and modifies that page "
                            "in place."
                        ),
                        "enum": ["create", "update"],
                        "default": "create",
                    },
                },
                "required": ["frontmatter"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=write_page,
    ),
    ToolDef(
        spec=types.Tool(
            name="write_pages",
            description=(
                "Batch-write a list of pages with a single indexer pass at "
                "the end. Same mode semantics as `write_page`: `create` "
                "mangles every id; `update` requires every id to exist "
                "already.\n\n"
                "Validate-all-then-act contract:\n"
                "  1. Every entry's frontmatter is validated up front.\n"
                "  2. For `mode='update'`, every entry's id is checked "
                "for existence before any writes happen. Missing id "
                "aborts the entire batch with `{error: 'not_found', "
                "index: N}`.\n"
                "  3. For `mode='create'`, no existence check is needed "
                "(mangling makes each id unique by construction).\n\n"
                "After all checks pass, writes proceed sequentially under "
                "the corpus mutex; the indexer runs once at the end."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "array",
                        "description": "List of `{frontmatter, body?}` entries.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "frontmatter": {"type": "object"},
                                "body": {"type": "string"},
                            },
                            "required": ["frontmatter"],
                        },
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "Write mode applied uniformly to every entry. "
                            "`create` (default) mangles each id; "
                            "`update` requires each id to already exist."
                        ),
                        "enum": ["create", "update"],
                        "default": "create",
                    },
                },
                "required": ["pages"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=write_pages,
    ),
    ToolDef(
        spec=types.Tool(
            name="add_link",
            description=(
                "Append an outgoing link to a page's `links_out` list. "
                "Read-modify-write under the corpus mutex: reads the page "
                "from disk, appends the link, writes back atomically, runs "
                "the incremental indexer. Duplicate links (same `target` "
                "and `label`) are detected and skipped — the response shape "
                "`{added: false, reason: 'duplicate'}` makes this explicit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "Id of the page the edge originates from.",
                    },
                    "to_id": {
                        "type": "string",
                        "description": "Id of the target page (or wiki path).",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional edge label (e.g. 'subdomain_of', 'cites', 'example_of').",
                    },
                    "via_source": {
                        "type": "string",
                        "description": "Optional source-id this edge was derived from.",
                    },
                },
                "required": ["from_id", "to_id"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=add_link,
    ),
    ToolDef(
        spec=types.Tool(
            name="add_claim",
            description=(
                "Append a Claim to a page's `claims` list. The claim "
                "object is validated against the `Claim` schema "
                "(id + text required; value, value_type, unit, confidence, "
                "source_ref optional). Read-modify-write under the corpus "
                "mutex. Duplicate claim ids are detected and skipped: "
                "response is `{added: false, reason: 'duplicate_claim_id'}`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Id of the page to append the claim to.",
                    },
                    "claim": {
                        "type": "object",
                        "description": (
                            "Claim shape: required `id` and `text`; optional "
                            "`value_type`, `value`, `unit`, `confidence` (0..1), "
                            "`confidence_label` (high/medium/low/unrated), "
                            "`source_ref`."
                        ),
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "value_type": {
                                "type": "string",
                                "enum": ["string", "number", "bool", "date"],
                            },
                            "value": {},
                            "unit": {"type": "string"},
                            "confidence": {"type": "number"},
                            "confidence_label": {
                                "type": "string",
                                "enum": ["high", "medium", "low", "unrated"],
                            },
                            "source_ref": {"type": "string"},
                        },
                        "required": ["id", "text"],
                    },
                },
                "required": ["page_id", "claim"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=add_claim,
    ),
    ToolDef(
        spec=types.Tool(
            name="add_links",
            description=(
                "Batch-append multiple outgoing links to one page's "
                "`links_out`. One disk read, one disk write, one indexer "
                "pass per call — versus N round-trips for callers that "
                "have many links to add (M3 ingest's entity-resolution "
                "stage being the motivating case).\n\n"
                "Validate-all-then-act: every link is validated against "
                "the `Link` schema before any disk work. Any validation "
                "failure aborts the whole batch with "
                "`{error: 'validation_error', index: N}`. Duplicate "
                "links (same `target` AND `label`) are NOT errors — "
                "they're reported per-item as "
                "`{added: false, reason: 'duplicate'}` (matches "
                "single-call `add_link`). Duplicate detection runs "
                "against existing links on disk AND against earlier "
                "items in the same batch.\n\n"
                "If 0 items actually need adding (all duplicates), the "
                "write + indexer pass is skipped entirely; "
                "`index_result` in the response is `null`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Id of the page to append links to.",
                    },
                    "links": {
                        "type": "array",
                        "description": "List of Link objects to append.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "label": {"type": "string"},
                                "via_source": {"type": "string"},
                            },
                            "required": ["target"],
                        },
                    },
                },
                "required": ["page_id", "links"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=add_links,
    ),
    ToolDef(
        spec=types.Tool(
            name="add_claims",
            description=(
                "Batch-append multiple Claims to one page's `claims` "
                "list. Same shape as `add_links`: one disk read, one "
                "disk write, one indexer pass.\n\n"
                "Validate-all-then-act: each claim is validated against "
                "the `Claim` schema; any validation failure aborts the "
                "whole batch. Duplicate claim ids (in existing claims OR "
                "in earlier batch items) are reported per-item as "
                "`{added: false, reason: 'duplicate_claim_id'}` — not "
                "errors. If 0 items need adding, the write + indexer "
                "pass is skipped."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Id of the page to append claims to.",
                    },
                    "claims": {
                        "type": "array",
                        "description": "List of Claim objects to append.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "value_type": {
                                    "type": "string",
                                    "enum": ["string", "number", "bool", "date"],
                                },
                                "value": {},
                                "unit": {"type": "string"},
                                "confidence": {"type": "number"},
                                "confidence_label": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low", "unrated"],
                                },
                                "source_ref": {"type": "string"},
                            },
                            "required": ["id", "text"],
                        },
                    },
                },
                "required": ["page_id", "claims"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=add_claims,
    ),
    ToolDef(
        spec=types.Tool(
            name="write_batch",
            description=(
                "Mixed-op atomic transaction: pages + links + claims + "
                "claim-updates in one MCP call. Single indexer pass at "
                "the end.\n\n"
                "Each op is one of:\n"
                "  - `{kind: 'write_page', frontmatter, body?, mode?}`\n"
                "  - `{kind: 'add_link', from_id, to_id, label?, via_source?}`\n"
                "  - `{kind: 'add_claim', page_id, claim}`\n"
                "  - `{kind: 'update_claim', page_id, claim_id, new_claim}`\n\n"
                "Three-phase contract (matches `write_pages`):\n"
                "  1. **Validate-all**: every op's kind + args validated "
                "before any disk work. Any failure aborts with "
                "`{error: 'validation_error'|'unknown_kind', index: N}`.\n"
                "  2. **Existence-check** (under corpus mutex): ops that "
                "reference an existing page verify it exists in LanceDB. "
                "**Cross-op references inside a single batch are NOT "
                "supported** — an `add_link` targeting a page CREATED "
                "earlier in the same batch will be reported as not_found. "
                "To create + reference, use two consecutive batches.\n"
                "  3. **Commit** (still under mutex): each op executes "
                "inline (same logic as its single-call counterpart). "
                "Multiple ops on the same page → multiple "
                "read-modify-writes (correct; intermediate state is "
                "mutex-protected). Indexer runs ONCE at the end.\n\n"
                "Destructive op kinds (`remove_*`) are not accepted in "
                "v1 — they're at the REMOVE_DESTRUCTIVE tier. Bulk op "
                "kinds (`add_links`, `add_claims`) aren't accepted either "
                "— callers flatten them into many single-op entries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ops": {
                        "type": "array",
                        "description": "List of op objects (see description for shapes).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "write_page",
                                        "add_link",
                                        "add_claim",
                                        "update_claim",
                                    ],
                                },
                            },
                            "required": ["kind"],
                        },
                    },
                },
                "required": ["ops"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=write_batch,
    ),
    ToolDef(
        spec=types.Tool(
            name="reindex_page",
            description=(
                "Force-re-index a single page from disk.\n\n"
                "Locates the page (LanceDB lookup first; filesystem "
                "walk fallback for pages that exist on disk but aren't "
                "in the index — e.g., post-restore), re-parses, "
                "projects to pages + embeddings + links + claims, "
                "refreshes FTS + ANN. Returns the per-page indexer "
                "summary.\n\n"
                "Use cases:\n"
                "  - The page was edited on disk outside the MCP write "
                "path.\n"
                "  - A high-stakes apply just landed and the operator "
                "wants belt-and-suspenders re-embedding.\n"
                "  - The page is on disk but not yet indexed.\n\n"
                "Runs under the corpus mutex; one indexer pass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Canonical id of the page to re-index.",
                    },
                },
                "required": ["page_id"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=reindex_page,
    ),
    ToolDef(
        spec=types.Tool(
            name="reindex_all",
            description=(
                "Wipe every LanceDB table, recreate schemas, walk "
                "`pages/`, project everything from disk. The explicit "
                "version of `bootstrap` for the re-init case — operators "
                "after a Restic restore (where `index/lance/` was "
                "excluded from the backup) use this to rebuild the "
                "derived index from the canonical markdown.\n\n"
                "**Async (C-13)**: returns immediately with a "
                "`task_id`; the actual work runs in the background "
                "under the corpus mutex. Poll `task_status(task_id)` "
                "to track progress; the final IndexResult appears in "
                "`task.result` when state reaches `succeeded`. "
                "Cancel via `task_cancel(task_id)` — cooperative, "
                "honored at safe boundaries (between indexer "
                "iterations and pre/post wipe).\n\n"
                "Response shape: `{task_id, kind: 'reindex_all', "
                "state: 'pending', created_at, message}`. "
                "Pre-C-13 callers expecting `wiped_tables` / "
                "`recreated_tables` / `index_result` directly in the "
                "response should switch to reading those fields from "
                "`task.result` after polling to terminal state."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=reindex_all,
    ),
    ToolDef(
        spec=types.Tool(
            name="task_cancel",
            description=(
                "Request cancellation of a scheduled task by "
                "`task_id` (C-13 async task model). Cooperative — "
                "the work function must check `task.check_cancel()` "
                "at safe boundaries to honor the request.\n\n"
                "Behavior by current state:\n"
                "  - PENDING task → transitions straight to "
                "CANCELLED; never runs.\n"
                "  - RUNNING task → `cancel_requested` flag set; "
                "work function bails at its next safe boundary. The "
                "response reflects state at the cancel-request "
                "moment — poll `task_status` to see the final "
                "CANCELLED transition.\n"
                "  - Terminal task → no-op; returns the existing "
                "task unchanged.\n\n"
                "Response: the full task dict (post-cancel-request "
                "state) plus `was_terminal_at_call: bool` "
                "indicating whether the cancel arrived too late.\n\n"
                "Errors: `not_found` if `task_id` is unknown."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task to cancel.",
                    },
                },
                "required": ["task_id"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=task_cancel,
    ),
    # ---- REMOVE_DESTRUCTIVE ----
    ToolDef(
        spec=types.Tool(
            name="remove_page",
            description=(
                "Cascading delete of a page by canonical id. Removes:\n"
                "  - the `.md` file from disk\n"
                "  - the `pages` row\n"
                "  - the `embeddings` row\n"
                "  - every outgoing link (from_id match)\n"
                "  - every incoming link (to_id match) — references to "
                "the gone page are silently dropped\n"
                "  - every claim attached to the page\n\n"
                "Use `incoming_links(page_id)` first to audit what "
                "references will be dropped. No alias resolution: pass "
                "the canonical id (use `find_by_alias` if you only have "
                "an alias)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Canonical id of the page to remove.",
                    },
                },
                "required": ["page_id"],
            },
        ),
        scope=Scope.REMOVE_DESTRUCTIVE,
        handler=remove_page,
    ),
    ToolDef(
        spec=types.Tool(
            name="update_claim",
            description=(
                "Replace one claim on a page, identified by `claim_id` "
                "within the page's `claims` list. The `new_claim` is "
                "validated against the `Claim` schema; its `id` must "
                "equal `claim_id` (no renaming via update — use "
                "add_claim + remove_claim if you need to rename). "
                "Read-modify-write under the corpus mutex."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "new_claim": {
                        "type": "object",
                        "description": "The replacement claim shape (Claim schema).",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "value_type": {
                                "type": "string",
                                "enum": ["string", "number", "bool", "date"],
                            },
                            "value": {},
                            "unit": {"type": "string"},
                            "confidence": {"type": "number"},
                            "confidence_label": {
                                "type": "string",
                                "enum": ["high", "medium", "low", "unrated"],
                            },
                            "source_ref": {"type": "string"},
                        },
                        "required": ["id", "text"],
                    },
                },
                "required": ["page_id", "claim_id", "new_claim"],
            },
        ),
        scope=Scope.REMOVE_DESTRUCTIVE,
        handler=update_claim,
    ),
    ToolDef(
        spec=types.Tool(
            name="remove_claim",
            description=(
                "Remove one claim from a page by `claim_id`. "
                "Read-modify-write under the corpus mutex. Returns "
                "`{error: 'claim_not_found'}` if the id isn't on the "
                "page."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "claim_id": {"type": "string"},
                },
                "required": ["page_id", "claim_id"],
            },
        ),
        scope=Scope.REMOVE_DESTRUCTIVE,
        handler=remove_claim,
    ),
    ToolDef(
        spec=types.Tool(
            name="remove_link",
            description=(
                "Remove an outgoing link from a page, matched by "
                "`(target, label)`. If `label` is omitted, removes "
                "EVERY edge from `from_id` to `to_id` regardless of "
                "label (returns a count). Otherwise removes only edges "
                "with matching `(target, label)`. RMW under the mutex. "
                "Returns `{removed: <count>}`; 0 means no matching link "
                "existed (not an error — symmetric with add_link's "
                "duplicate no-op)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "Id of the page the link originates from.",
                    },
                    "to_id": {
                        "type": "string",
                        "description": "Id of the target page (the link's `target`).",
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Optional edge label. Omit to remove every "
                            "edge between from_id and to_id regardless "
                            "of label."
                        ),
                    },
                },
                "required": ["from_id", "to_id"],
            },
        ),
        scope=Scope.REMOVE_DESTRUCTIVE,
        handler=remove_link,
    ),
]


_TOOLS_BY_NAME: dict[str, ToolDef] = {t.spec.name: t for t in TOOLS}


# ---- listing + dispatch ----


def list_tools(scope: Scope) -> list[types.Tool]:
    """Return the tool specs the caller is allowed to see.

    Tier-based: caller at tier N sees every tool whose required scope is ≤ N.
    """
    caller_tier = SCOPE_TIER[scope]
    return [t.spec for t in TOOLS if SCOPE_TIER[t.scope] <= caller_tier]


async def dispatch(name: str, arguments: dict[str, Any], *, app: App, scope: Scope) -> dict[str, Any]:
    """Run a tool by name. Raises if the tool is unknown or the scope is insufficient."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        raise KeyError(f"unknown tool: {name}")
    if SCOPE_TIER[tool.scope] > SCOPE_TIER[scope]:
        raise PermissionError(
            f"tool {name!r} requires scope {tool.scope.value!r}; caller has {scope.value!r}"
        )
    return await tool.handler(app, arguments)
