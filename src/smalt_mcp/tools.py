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

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp import types

from smalt_mcp.permissions import Scope
from smalt_mcp.storage import lance

if TYPE_CHECKING:
    from smalt_mcp.app import App


logger = logging.getLogger(__name__)


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
    """List indexed pages, optionally filtered by `type` and/or `prefix` on id."""
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    type_filter = arguments.get("type")
    prefix = arguments.get("prefix")
    limit = int(arguments.get("limit", 100))

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)

    where_parts: list[str] = []
    if type_filter:
        where_parts.append(f"type = {lance.sql_str(type_filter)}")
    if prefix:
        where_parts.append(f"id LIKE {lance.sql_str(prefix + '%')}")

    query = pages.search().select(["id", "title", "type", "path"])
    if where_parts:
        query = query.where(" AND ".join(where_parts))
    arrow = query.limit(limit).to_arrow()

    ids = arrow.column("id").to_pylist()
    titles = arrow.column("title").to_pylist()
    types_ = arrow.column("type").to_pylist()
    paths = arrow.column("path").to_pylist()

    out_pages = [
        {"id": pid, "title": t, "type": tp, "path": p}
        for pid, t, tp, p in zip(ids, titles, types_, paths, strict=True)
    ]
    return {"pages": out_pages, "count": len(out_pages)}


# ---- handler: read_page ----


async def read_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the frontmatter (parsed) + body of a single page by id."""
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    page_id = arguments.get("page_id")
    if not page_id:
        return {"error": "missing_argument", "message": "page_id is required"}

    db = app.db()
    pages = db.open_table(lance.TABLE_PAGES)
    arrow = (
        pages.search()
        .where(f"id = {lance.sql_str(page_id)}")
        .select(["id", "title", "type", "path", "body", "frontmatter_json"])
        .limit(1)
        .to_arrow()
    )
    if arrow.num_rows == 0:
        return {"error": "not_found", "page_id": page_id}

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


# ---- handler: traverse ----


async def traverse(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """1-hop graph traversal from `from_id` via the links table.

    Returns the outgoing edges; optionally filtered by `label`. `hops > 1` is
    accepted but not yet implemented; the v0 surface returns 1-hop only.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    from_id = arguments.get("from_id")
    if not from_id:
        return {"error": "missing_argument", "message": "from_id is required"}
    label = arguments.get("label")
    hops = int(arguments.get("hops", 1))
    if hops > 1:
        logger.info("traverse: hops=%s requested but v0 returns 1-hop only", hops)

    db = app.db()
    links = db.open_table(lance.TABLE_LINKS)

    where = f"from_id = {lance.sql_str(from_id)}"
    if label:
        where += f" AND label = {lance.sql_str(label)}"

    arrow = (
        links.search()
        .where(where)
        .select(["from_id", "to_id", "label"])
        .limit(1000)
        .to_arrow()
    )
    edges = [
        {"from_id": f, "to_id": t, "label": lbl}
        for f, t, lbl in zip(
            arrow.column("from_id").to_pylist(),
            arrow.column("to_id").to_pylist(),
            arrow.column("label").to_pylist(),
            strict=True,
        )
    ]
    return {"from_id": from_id, "edges": edges, "count": len(edges)}


# ---- handler: search ----


def _rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of N ranked lists. Returns (id, score) sorted desc."""
    scores: dict[str, float] = {}
    for ranks in rankings:
        for rank, pid in enumerate(ranks, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


async def search(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Hybrid search over the pages corpus: FTS (body) + vector (embeddings), RRF-fused.

    Returns the top-`top_k` matches with id, title, type, snippet, score. If
    the FTS index hasn't been built yet (very small Smalt), falls back to
    vector-only ranking.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    query = arguments.get("query")
    if not query:
        return {"error": "missing_argument", "message": "query is required"}
    top_k = int(arguments.get("top_k", 10))
    fetch_k = max(top_k * 3, top_k + 5)  # over-fetch for RRF; clamp at top_k below

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

    if not fts_ids and not vec_ids:
        return {"results": [], "count": 0}

    fused = _rrf_fuse([fts_ids, vec_ids])
    top = fused[:top_k]
    top_ids = [pid for pid, _ in top]

    # Hydrate with page metadata in one query.
    quoted = ", ".join(lance.sql_str(p) for p in top_ids)
    meta_arrow = (
        pages.search()
        .where(f"id IN ({quoted})")
        .select(["id", "title", "type", "body"])
        .limit(len(top_ids))
        .to_arrow()
    )
    by_id: dict[str, dict[str, Any]] = {}
    for i in range(meta_arrow.num_rows):
        pid = meta_arrow.column("id")[i].as_py()
        by_id[pid] = {
            "title": meta_arrow.column("title")[i].as_py(),
            "type": meta_arrow.column("type")[i].as_py(),
            "body": meta_arrow.column("body")[i].as_py() or "",
        }

    results = []
    for pid, score in top:
        meta = by_id.get(pid)
        if not meta:
            continue
        body = meta["body"]
        snippet = body[:200] + ("…" if len(body) > 200 else "")
        results.append(
            {
                "id": pid,
                "title": meta["title"],
                "type": meta["type"],
                "snippet": snippet,
                "score": round(score, 6),
            }
        )
    return {"results": results, "count": len(results)}


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
                "List indexed pages in the Smalt, optionally filtered by `type` "
                "(entity/concept/source/synthesis) and/or `prefix` (id starts "
                "with). Returns minimal metadata per page (id, title, type, "
                "path) — use `read_page` to fetch a single page's full body + "
                "frontmatter."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Optional page-type filter.",
                        "enum": ["entity", "concept", "source", "synthesis"],
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
                "Return one page's full body + parsed frontmatter, looked up "
                "by id. Returns `{error: 'not_found'}` if the id isn't indexed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The page id, e.g. 'ent-alice' or 'con-embedding'.",
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
            name="traverse",
            description=(
                "Walk outgoing links from a page. Returns the edges "
                "(from_id, to_id, label) that originate at `from_id`, "
                "optionally filtered by edge `label`. v0 returns 1-hop only; "
                "`hops > 1` is accepted but currently ignored."
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
                        "description": "Optional edge-label filter.",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of hops to walk. v0 only honors 1.",
                        "default": 1,
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
            name="search",
            description=(
                "Hybrid search over the Smalt's pages: FTS (body) + vector "
                "(summary embedding), fused via Reciprocal Rank Fusion. "
                "Returns top-`top_k` matches with id, title, type, snippet, "
                "and an RRF score. If the FTS index isn't built yet (very "
                "small Smalts), falls back to vector-only ranking."
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
                },
                "required": ["query"],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=search,
    ),
]


_TOOLS_BY_NAME: dict[str, ToolDef] = {t.spec.name: t for t in TOOLS}


# ---- listing + dispatch ----


def list_tools(scope: Scope) -> list[types.Tool]:
    """Return the tool specs the caller is allowed to see."""
    if scope is Scope.READ_WRITE:
        return [t.spec for t in TOOLS]
    return [t.spec for t in TOOLS if t.scope is Scope.READ_ONLY]


async def dispatch(name: str, arguments: dict[str, Any], *, app: App, scope: Scope) -> dict[str, Any]:
    """Run a tool by name. Raises if the tool is unknown or the scope is insufficient."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        raise KeyError(f"unknown tool: {name}")
    if scope is Scope.READ_ONLY and tool.scope is Scope.READ_WRITE:
        raise PermissionError(f"tool {name!r} requires read-write scope")
    return await tool.handler(app, arguments)
