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

import base64
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter
from mcp import types
from pydantic import TypeAdapter, ValidationError

from smalt_mcp.permissions import SCOPE_TIER, Scope
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


# ---- shared helpers ----


def _not_initialized() -> dict[str, Any]:
    return {"error": "smalt_not_initialized", "message": "Smalt directory or LanceDB tables not present; bootstrap first."}


def _ensure_initialized(app: App) -> tuple[bool, dict[str, Any] | None]:
    """Return (ok, error_payload). Use as: `ok, err = _ensure_initialized(app); if not ok: return err`."""
    if not app.smalt_exists():
        return False, _not_initialized()
    try:
        app.db()
    except FileNotFoundError:
        return False, _not_initialized()
    return True, None


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
    """Compute the canonical on-disk path for `page` inside `smalt_root`."""
    subdir = _TYPE_TO_SUBDIR[page.type]
    return paths.pages_dir(smalt_root) / subdir / f"{page.id}.md"


def _run_indexer(app: App) -> dict[str, Any]:
    """Run an incremental indexer pass. Caller must hold the corpus mutex."""
    from smalt_mcp.storage.indexer import Indexer

    result = Indexer(
        smalt_root=app.cfg.smalt_dir,
        embedder=app.embedder(),
        db=app.db(),
    ).run()
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

_PROPERTY_FILTER_ARGS: frozenset[str] = frozenset({
    "glossary",
    "is_domain",
    "domain",
    "fetched_at_before",
    "fetched_at_after",
    "has_aliases_containing",
})


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


async def status(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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


# ---- handler: list_pages ----


async def list_pages(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List indexed pages, optionally filtered by `type` / `prefix` (LanceDB-side)
    and/or any of the property filters (`glossary`, `is_domain`, `domain`,
    `fetched_at_before`, `fetched_at_after`, `has_aliases_containing`,
    client-side post-fetch). All filters AND-composed; `limit` applies to the
    final filtered set."""
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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

    Scans the pages table's `frontmatter_json` column — no LanceDB-side
    index on aliases today, so this is O(pages). Fine for typical Smalts
    (thousands of pages); when we hit limits we'll add an aliases table.
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = (
        pages.search()
        .select(["id", "title", "type", "path", "frontmatter_json"])
        .to_arrow()
    )
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


async def read_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the frontmatter (parsed) + body of a single page.

    Lookup order:
      1. Exact match on the canonical `id`. If found, return.
      2. Alias fallback: search every page's `aliases` for `page_id`.
         - Exactly one match → return that page, with `resolved_via_alias: true`.
         - Two or more matches → `{error: 'ambiguous_alias', matches: [...]}`.
         - Zero matches → `{error: 'not_found', page_id: ...}`.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}

    # 1. Exact id match
    payload = _fetch_page_row(app, page_id)
    if payload is not None:
        return payload

    # 2. Alias fallback
    matches = _find_pages_by_alias(app, page_id)
    if not matches:
        return {"error": "not_found", "page_id": page_id}
    if len(matches) > 1:
        return {
            "error": "ambiguous_alias",
            "alias": page_id,
            "matches": matches,
            "message": (
                f"alias {page_id!r} matches {len(matches)} pages; "
                "address by canonical id (use the `id` field of one of the matches above)"
            ),
        }
    # Exactly one match — fetch the full row.
    canonical = matches[0]["id"]
    payload = _fetch_page_row(app, canonical)
    if payload is None:  # shouldn't happen — index just told us this exists
        return {"error": "not_found", "page_id": canonical}
    payload["resolved_via_alias"] = page_id
    return payload


# ---- handler: find_by_alias ----


async def find_by_alias(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List every page whose `aliases` contains `alias`.

    Use this when you have a memorable handle (the original caller-id
    before mangling, or any hand-added alias) and want to find the page(s)
    it maps to. Returns minimal metadata (id, title, type, path) per match
    — call `read_page` with the canonical `id` to get the body.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    alias = arguments.get("alias")
    if not alias:
        return {"error": "missing_argument", "message": "alias is required"}

    matches = _find_pages_by_alias(app, alias)
    return {"alias": alias, "matches": matches, "count": len(matches)}


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


async def traverse(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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


async def incoming_links(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List every link whose `to_id` matches `page_id` — the "what points
    at me" view.

    Symmetric to `traverse` (which lists OUTGOING links). Useful to audit
    references before calling `remove_page` (which cascades — removing a
    page silently drops all incoming references; this tool lets the caller
    see them first).
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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

    Returns the page_ids in `pages`-table-scan order. The downstream RRF
    treats them as a single ranked list — pages that match earlier in
    the scan get slightly higher rank, but the ranking signal is weak.
    The point is presence in the input set, not relative ordering inside it.
    """
    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = (
        pages.search()
        .select(["id", "frontmatter_json"])
        .to_arrow()
    )
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


async def search(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
        fetch_k = max(fetch_k * 5, 100)  # 5× extra room for filter dropoff

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)

    # FTS over body (and title; LanceDB FTS indexes one field per call, we
    # have one for each — search uses whichever the table considers primary).
    fts_ids: list[str] = []
    try:
        fts_arrow = (
            pages.search(query, query_type="fts")
            .select(["id"])
            .limit(fetch_k)
            .to_arrow()
        )
        fts_ids = fts_arrow.column("id").to_pylist()
    except Exception as e:  # noqa: BLE001 — FTS index might not be built yet
        logger.info("FTS search unavailable (%s); falling back to vector-only", e)

    # Vector over embeddings table; pull page_ids ranked by similarity.
    vec = app.embedder().embed([query])[0]
    embs = db.open_table(lance.TABLE_EMBEDDINGS)
    try:
        vec_arrow = (
            embs.search(vec, vector_column_name="vector")
            .select(["page_id"])
            .limit(fetch_k)
            .to_arrow()
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
    if has_props:
        candidate_pool = fused[:fetch_k]
    else:
        candidate_pool = fused[:top_k]
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


async def list_domains(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List ConceptPages flagged `is_domain: true`.

    Domain hierarchy itself (which domain is a subdomain of which) lives in
    each domain's `subdomain_of` labeled links in `links_out`, NOT in this
    response. Use `traverse(from_id=<domain>, label='subdomain_of')` to walk
    the hierarchy.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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


# ---- handler: bootstrap (READ_WRITE) ----


async def bootstrap(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    arrow = (
        pages.search()
        .where(f"id = {lance.sql_str(page_id)}")
        .select(["path"])
        .limit(1)
        .to_arrow()
    )
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


async def write_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
        # Mangle and re-validate (re-validation is cheap and confirms the
        # canonical id still passes _validate_id — URL-safe base64 is in the
        # allowed character set).
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


async def write_pages(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
    arrow = (
        pages.search()
        .where(f"id = {lance.sql_str(page_id)}")
        .select(["path"])
        .limit(1)
        .to_arrow()
    )
    if arrow.num_rows == 0:
        return None
    rel = arrow.column("path")[0].as_py()
    return app.cfg.smalt_dir / rel


async def add_link(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append an outgoing link to a page's `links_out` (read-modify-write).

    Locates the page by id, reads its current frontmatter from disk (not
    LanceDB — we want the latest), appends `{target, label?, via_source?}` to
    `links_out`, writes back atomically, and runs an incremental indexer
    pass. Skips duplicates: a link with the same `target` AND `label`
    already in the list is a no-op and `added: false, reason: duplicate`.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
        new_fm["links_out"] = existing_links + [new_link]

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


async def add_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Append a Claim to a page's `claims` list (read-modify-write).

    Locates the page by id, validates the claim against the `Claim` Pydantic
    model, then reads the page's current frontmatter from disk, appends the
    raw claim dict to `claims`, writes back atomically, and runs the
    indexer. Skips duplicates: a claim with an id already present in the
    list is a no-op and `added: false, reason: duplicate_claim_id`.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
        new_fm["claims"] = existing_claims + [claim]

        _serialize_and_write_page(page_path, new_fm, parsed.body)
        index_result = _run_indexer(app)

    return {
        "id": page_id,
        "added": True,
        "claim_id": validated_claim.id,
        "claims_count": len(new_fm["claims"]),
        "index_result": index_result,
    }


# ---- handler: remove_page (REMOVE_DESTRUCTIVE) ----


async def remove_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
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
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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


async def update_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace one claim on a page, identified by `claim_id` within the
    page's `claims` list. Read-modify-write under the corpus mutex.

    The new claim is validated against the `Claim` schema. The claim id
    in `new_claim` must match `claim_id` (we don't let updates change the
    identifier — use add_claim + remove_claim if you want to rename).
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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


async def remove_claim(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Remove one claim from a page by `claim_id`. RMW under the mutex.

    Returns `{error: 'claim_not_found'}` if the claim id isn't on the page.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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


async def remove_link(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Remove an outgoing link from a page, matched by (target, label).

    If `label` is omitted, removes EVERY edge from `from_id` to `to_id`
    regardless of label (returns a count). Otherwise removes only edges
    with matching `(target, label)`.

    RMW under the mutex. Returns `{removed: <count>}`; 0 means no matching
    link existed (not an error — symmetric with add_link's duplicate-no-op).
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

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
                "  2. Alias fallback: if no exact match, search every "
                "page's `aliases` for `page_id`.\n"
                "      - 1 match → return that page; response includes "
                "`resolved_via_alias: <the alias>`.\n"
                "      - 2+ matches → `{error: 'ambiguous_alias', "
                "matches: [...]}` (caller must pick a canonical id).\n"
                "      - 0 matches → `{error: 'not_found'}`.\n\n"
                "Use `find_by_alias` if you want the full match list "
                "without picking one."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": (
                            "Canonical page id (e.g. 'ent-alice-XYZ…') or a "
                            "known alias (e.g. 'ent-alice'). Exact-id "
                            "lookup runs first; alias fallback is "
                            "automatic."
                        ),
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
                "to."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "The alias to look up.",
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
                            "Optional edge-label filter, applied per hop "
                            "(only matching edges are followed)."
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
