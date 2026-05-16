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
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter
from mcp import types
from pydantic import TypeAdapter, ValidationError

from smalt_mcp.permissions import Scope
from smalt_mcp.schema import Claim, Page, PageType, ProposalKind, ProposalPage
from smalt_mcp.storage import lance, paths
from smalt_mcp.storage.markdown import parse_page

if TYPE_CHECKING:
    from smalt_mcp.app import App


logger = logging.getLogger(__name__)


PAGE_ADAPTER: TypeAdapter[Page] = TypeAdapter(Page)
PROPOSAL_ADAPTER: TypeAdapter[ProposalPage] = TypeAdapter(ProposalPage)


# Proposal kinds that go to `tasks/proposals/schema/` regardless of who proposed them.
_SCHEMA_PROPOSAL_KINDS: frozenset[ProposalKind] = frozenset(
    {
        ProposalKind.SCHEMA_ADDITION,
        ProposalKind.SCHEMA_DRIFT,
        ProposalKind.SCHEMA_REMOVAL,
    }
)


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

This document is a **living artifact** — proposals, audits, and edits flow
through the same `tasks/proposals/` queue as everything else.
"""

_POLICY_MD_PLACEHOLDER = """# POLICY.md

This is the human-readable policy that agentic systems operate under when
producing or modifying pages in this Smalt: when to create new pages vs.
extend, how contradictions are handled, how confidence is assigned, and
the falsifiability / cost-tier rules from the proposal-as-hypothesis
discipline.

Like SCHEMA.md, this document is **living** — edits flow through the
proposal queue.
"""

_GAPS_MD_PLACEHOLDER = """# Knowledge gaps

Gap signals emitted by the retrieve and converse systems land here. Each
entry is one knowledge gap; the research system reads this queue and
proposes sources to fill them.
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


# ---- handler: list_proposals (READ_ONLY) ----


async def list_proposals(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """List ProposalPages in `tasks/proposals/`, optionally filtered.

    Proposals are NOT indexed in LanceDB — they live in the filesystem under
    `tasks/proposals/<system or schema>/<id>.md`. This handler walks that
    tree, parses frontmatter, applies the filters, and returns minimal
    metadata per match.
    """
    if not app.smalt_exists():
        return _not_initialized()

    system = arguments.get("system")  # subdir match: cogitate / curate / research / schema / etc.
    status = arguments.get("status")  # proposal lifecycle state
    kind = arguments.get("kind")      # proposal_kind

    proposals_root = paths.proposals_dir(app.cfg.smalt_dir)
    if not proposals_root.exists():
        return {"proposals": [], "count": 0}

    import frontmatter as _fm  # local alias; module-level `frontmatter` is the python-frontmatter import

    out: list[dict[str, Any]] = []
    for f in sorted(proposals_root.rglob("*.md")):
        if not f.is_file():
            continue
        rel = f.relative_to(proposals_root)
        # System filter: subdir name
        if system and (len(rel.parts) < 2 or rel.parts[0] != system):
            continue
        try:
            post = _fm.load(str(f))
        except Exception:  # noqa: BLE001 — skip unparseable
            continue

        # Resolve schema defaults at read time: if validation succeeds, use the
        # validated model's effective values (status, test_status, test_cost
        # etc. get their defaults). If a proposal is malformed, fall back to
        # raw frontmatter so we don't silently drop it from listings.
        md = post.metadata
        try:
            proposal = PROPOSAL_ADAPTER.validate_python(md)
            eff_kind = proposal.proposal_kind.value
            eff_status = proposal.status.value
            eff_proposed_by = proposal.proposed_by
            eff_id = proposal.id
            eff_title = proposal.title
            eff_proposed_at = proposal.proposed_at.isoformat()
        except ValidationError:
            eff_kind = md.get("proposal_kind")
            eff_status = md.get("status")
            eff_proposed_by = md.get("proposed_by")
            eff_id = md.get("id")
            eff_title = md.get("title")
            eff_proposed_at = md.get("proposed_at")

        if status and eff_status != status:
            continue
        if kind and eff_kind != kind:
            continue

        out.append(
            {
                "id": eff_id,
                "title": eff_title,
                "proposal_kind": eff_kind,
                "status": eff_status,
                "proposed_by": eff_proposed_by,
                "proposed_at": eff_proposed_at,
                "path": str(f.relative_to(app.cfg.smalt_dir)),
                "subdir": rel.parts[0] if len(rel.parts) >= 2 else "",
            }
        )
    return {"proposals": out, "count": len(out)}


# ---- handler: bootstrap (READ_WRITE) ----


async def bootstrap(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Initialize an empty Smalt at the configured `SMALT_DIR`.

    Creates the canonical directory layout, drops in SCHEMA.md / POLICY.md /
    tasks/gaps.md placeholders if they're missing, and creates the LanceDB
    tables. Idempotent: existing directories / files / tables are left alone;
    the response reports only what was *newly* created.
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
        ("tasks/gaps.md", _GAPS_MD_PLACEHOLDER),
    ):
        target = smalt_root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            created_files.append(rel)

    created_tables = lance.ensure_tables(smalt_root, embedding_dim=app.cfg.embedding.dim)

    return {
        "smalt_dir": str(smalt_root),
        "created_dirs": created_dirs,
        "created_files": created_files,
        "created_tables": created_tables,
    }


# ---- write mode + existence helpers ----


_VALID_WRITE_MODES: frozenset[str] = frozenset({"create", "update", "upsert"})


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


def _check_mode_against_existence(
    mode: str, page_id: str, existing_path: str | None
) -> dict[str, Any] | None:
    """Compare requested write mode against current existence; return an error
    payload if the mode contract is violated, else None."""
    if mode == "create" and existing_path is not None:
        return {
            "error": "already_exists",
            "id": page_id,
            "existing_path": existing_path,
            "message": f"page {page_id!r} already exists; use mode='update' or mode='upsert' to overwrite",
        }
    if mode == "update" and existing_path is None:
        return {
            "error": "not_found",
            "id": page_id,
            "message": f"page {page_id!r} does not exist; use mode='create' or mode='upsert' to insert",
        }
    return None


# ---- handler: write_page (READ_WRITE) ----


async def write_page(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Write one page (frontmatter + body) and trigger an incremental indexer pass.

    Path is derived from the page's type + id: `pages/<subdir>/<page_id>.md`.
    Atomic at the filesystem level (tmp-then-rename). Wrapped in the corpus
    mutex so the existence check + write + indexer pass form a single critical
    section.

    The `mode` arg controls whether the write is restricted:
      - "create" (strict insert): fail if the id already exists
      - "update" (strict update): fail if the id does NOT exist
      - "upsert" (default; backward compatible): always proceed

    Response includes `created: bool` so callers know which happened regardless
    of mode.
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    fm = arguments.get("frontmatter")
    if not fm:
        return {"error": "missing_argument", "message": "frontmatter is required"}
    body = arguments.get("body") or ""
    mode = arguments.get("mode") or "upsert"
    if mode not in _VALID_WRITE_MODES:
        return {
            "error": "invalid_argument",
            "message": f"mode must be one of {sorted(_VALID_WRITE_MODES)}; got {mode!r}",
        }

    try:
        page = PAGE_ADAPTER.validate_python(fm)
    except ValidationError as e:
        return {"error": "validation_error", "message": str(e)}

    target = _page_target_path(app.cfg.smalt_dir, page)

    with app.mutex.acquire("write_page"):
        existing_path = _existing_page_path(app, page.id)
        mode_err = _check_mode_against_existence(mode, page.id, existing_path)
        if mode_err is not None:
            return mode_err
        created = existing_path is None
        _serialize_and_write_page(target, fm, body)
        index_result = _run_indexer(app)

    return {
        "id": page.id,
        "path": str(target.relative_to(app.cfg.smalt_dir)),
        "type": page.type.value,
        "mode": mode,
        "created": created,
        "index_result": index_result,
    }


# ---- handler: write_pages (READ_WRITE) — batch ----


async def write_pages(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-write a list of pages with a single indexer pass at the end.

    Validate-all-then-write contract:
      1. Every page's frontmatter is validated against the Page Pydantic union.
      2. If a `mode` is set ("create" or "update"), every entry's id is checked
         against the existence table BEFORE any writes happen.
      3. If any check fails, no writes occur; the response reports the
         offending index.

    The `mode` arg applies to every entry uniformly; default is "upsert".
    """
    ok, err = _ensure_initialized(app)
    if not ok:
        return err  # type: ignore[return-value]

    items = arguments.get("pages")
    if not items or not isinstance(items, list):
        return {"error": "missing_argument", "message": "pages must be a non-empty list"}
    mode = arguments.get("mode") or "upsert"
    if mode not in _VALID_WRITE_MODES:
        return {
            "error": "invalid_argument",
            "message": f"mode must be one of {sorted(_VALID_WRITE_MODES)}; got {mode!r}",
        }

    # Phase 1 (no disk touch): validate every entry's frontmatter.
    validated: list[tuple[Page, dict[str, Any], str]] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            return {"error": "validation_error", "index": i, "message": "each entry must be an object"}
        fm = entry.get("frontmatter")
        if not fm:
            return {"error": "validation_error", "index": i, "message": "frontmatter is required"}
        body = entry.get("body") or ""
        try:
            page = PAGE_ADAPTER.validate_python(fm)
        except ValidationError as e:
            return {"error": "validation_error", "index": i, "message": str(e)}
        validated.append((page, fm, body))

    written: list[dict[str, Any]] = []
    with app.mutex.acquire("write_pages"):
        # Phase 2 (still no disk touch): mode-vs-existence checks, all-or-nothing.
        existences: list[str | None] = []
        for i, (page, _fm, _body) in enumerate(validated):
            existing_path = _existing_page_path(app, page.id)
            mode_err = _check_mode_against_existence(mode, page.id, existing_path)
            if mode_err is not None:
                # Annotate with which entry failed; abort the whole batch.
                mode_err["index"] = i
                return mode_err
            existences.append(existing_path)

        # Phase 3: commit. All checks passed; write every page.
        for (page, fm, body), existing_path in zip(validated, existences, strict=True):
            target = _page_target_path(app.cfg.smalt_dir, page)
            _serialize_and_write_page(target, fm, body)
            written.append(
                {
                    "id": page.id,
                    "path": str(target.relative_to(app.cfg.smalt_dir)),
                    "type": page.type.value,
                    "created": existing_path is None,
                }
            )
        index_result = _run_indexer(app)

    return {
        "written": written,
        "count": len(written),
        "mode": mode,
        "index_result": index_result,
    }


# ---- handler: write_proposal (READ_WRITE) ----


def _proposal_target_path(smalt_root: Path, proposal: ProposalPage) -> Path:
    """Route a proposal to its on-disk path.

    Schema-related kinds go to `tasks/proposals/schema/`; everything else
    goes to `tasks/proposals/<proposed_by>/`. Per `cobalt-grinding/docs/
    plan.md` → "Proposal document shape and lifecycle".
    """
    if proposal.proposal_kind in _SCHEMA_PROPOSAL_KINDS:
        subdir = "schema"
    else:
        subdir = proposal.proposed_by
    return paths.proposals_dir(smalt_root) / subdir / f"{proposal.id}.md"


async def write_proposal(app: App, arguments: dict[str, Any]) -> dict[str, Any]:
    """Write a ProposalPage to `tasks/proposals/<subdir>/<id>.md`.

    Subdir = `schema` for schema_addition / schema_drift / schema_removal
    kinds; otherwise = `proposed_by`. Atomic at the filesystem level
    (tmp-then-rename). Proposals are NOT projected into LanceDB — they're
    queryable via `list_proposals`.
    """
    if not app.smalt_exists():
        return _not_initialized()

    fm = arguments.get("frontmatter")
    if not fm:
        return {"error": "missing_argument", "message": "frontmatter is required"}
    body = arguments.get("body") or ""

    try:
        proposal = PROPOSAL_ADAPTER.validate_python(fm)
    except ValidationError as e:
        return {"error": "validation_error", "message": str(e)}

    target = _proposal_target_path(app.cfg.smalt_dir, proposal)

    # Proposals don't go through the corpus mutex — they're outside the
    # indexed pages/ tree. Atomic write is still required so a reader never
    # sees a half-written file.
    _serialize_and_write_page(target, fm, body)

    return {
        "id": proposal.id,
        "path": str(target.relative_to(app.cfg.smalt_dir)),
        "subdir": target.parent.name,
        "proposal_kind": proposal.proposal_kind.value,
        "status": proposal.status.value,
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
            name="list_proposals",
            description=(
                "List ProposalPages in `tasks/proposals/`, optionally "
                "filtered by `system` (subdir name — cogitate / curate / "
                "research / schema / toolsmith / converse), `status` "
                "(proposed / under_test / validated / rejected / applied / "
                "superseded), and/or `kind` (proposal_kind value)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                        "description": "Filter by subdir / proposing system.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by lifecycle state.",
                        "enum": [
                            "proposed",
                            "under_test",
                            "validated",
                            "rejected",
                            "applied",
                            "superseded",
                        ],
                    },
                    "kind": {
                        "type": "string",
                        "description": "Filter by proposal_kind (e.g. schema_addition, source_adoption).",
                    },
                },
                "required": [],
            },
        ),
        scope=Scope.READ_ONLY,
        handler=list_proposals,
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
                "Create or update one page. The frontmatter must conform to "
                "the Page Pydantic union (entity / concept / source / "
                "synthesis discriminated by the `type` field). The file is "
                "written atomically (tmp-then-rename) at "
                "`pages/<subdir>/<page_id>.md`, where <subdir> derives from "
                "the page's type. After the write, an incremental indexer "
                "pass refreshes the LanceDB tables; the response includes "
                "the indexer's summary. Both write and index run inside the "
                "corpus single-writer mutex.\n\n"
                "The `mode` arg restricts what writes are allowed:\n"
                "  - `create` (strict insert): fail with `{error: 'already_exists'}` "
                "if the id is already indexed.\n"
                "  - `update` (strict update): fail with `{error: 'not_found'}` "
                "if the id is not already indexed.\n"
                "  - `upsert` (default): always proceed; existing id overwrites.\n\n"
                "The response includes `created: bool` so callers know which "
                "happened regardless of mode."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "frontmatter": {
                        "type": "object",
                        "description": (
                            "Full page frontmatter. Required keys: id, type, "
                            "title. Other keys per the page-type schema "
                            "(see smalt_mcp/schema.py). Note: the id is "
                            "validated for path-safety + portability "
                            "(alphanumeric + underscore + hyphen, no leading "
                            "dash/underscore, 1..254 chars, not a Windows-"
                            "reserved name)."
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
                            "Write mode. `create` = strict insert (fail if id "
                            "exists); `update` = strict update (fail if id "
                            "doesn't exist); `upsert` = either."
                        ),
                        "enum": ["create", "update", "upsert"],
                        "default": "upsert",
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
                "the end. Validate-all-then-write contract:\n"
                "  1. Every page's frontmatter is validated up front.\n"
                "  2. If `mode` is `create` or `update`, every entry's id is "
                "checked against the index BEFORE any writes happen.\n"
                "  3. If any validation or mode check fails, no writes occur; "
                "the response reports the offending index.\n\n"
                "After all checks pass, writes proceed sequentially under the "
                "corpus mutex; the indexer runs once at the end."
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
                            "`create` = all must be new; `update` = all must "
                            "already exist; `upsert` = either."
                        ),
                        "enum": ["create", "update", "upsert"],
                        "default": "upsert",
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
            name="write_proposal",
            description=(
                "Write a ProposalPage to `tasks/proposals/<subdir>/<id>.md`. "
                "Subdir = `schema` for the schema-related proposal kinds "
                "(schema_addition / schema_drift / schema_removal); otherwise "
                "= `proposed_by` (cogitate / curate / research / toolsmith / "
                "converse). Atomic at the filesystem level. Proposals are NOT "
                "projected into LanceDB — they're queryable via "
                "`list_proposals`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "frontmatter": {
                        "type": "object",
                        "description": (
                            "ProposalPage frontmatter. Required keys: id, "
                            "type='proposal', title, proposal_kind, "
                            "proposed_by, proposed_at. Optional: status, "
                            "test_status, test_cost, related_pages, "
                            "supersedes, superseded_by."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Markdown body with the standard "
                            "Observation/Hypothesis/Prediction/Test/Reasoning "
                            "sections."
                        ),
                        "default": "",
                    },
                },
                "required": ["frontmatter"],
            },
        ),
        scope=Scope.READ_WRITE,
        handler=write_proposal,
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
