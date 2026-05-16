"""LanceDB connection + table-creation helpers.

This module *creates* the empty tables and exposes a connection helper plus
upsert/delete helpers used by the indexer. Tables are defined with explicit
Arrow schemas so they round-trip cleanly without an embedding model present.

The schemas mirror the LanceDB-tables list in `cobalt-grinding/docs/plan.md`.
Keep them in sync when the plan updates.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import lancedb

from smalt_mcp.storage.paths import lance_dir

# ---- table names ----

TABLE_PAGES = "pages"
TABLE_EMBEDDINGS = "embeddings"
TABLE_LINKS = "links"
TABLE_CLAIMS = "claims"
TABLE_SOURCES = "sources"

ALL_TABLES: tuple[str, ...] = (
    TABLE_PAGES,
    TABLE_EMBEDDINGS,
    TABLE_LINKS,
    TABLE_CLAIMS,
    TABLE_SOURCES,
)


# ---- arrow schemas ----
# Embedding-vector dim is configurable; we accept a runtime value rather than
# baking it in.


def pages_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
            pa.field("type", pa.string(), nullable=False),
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
            pa.field("frontmatter_json", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("updated_at", pa.timestamp("us", tz="UTC")),
        ]
    )


def embeddings_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("page_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("model_version", pa.string()),
        ]
    )


def links_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("from_id", pa.string(), nullable=False),
            pa.field("to_id", pa.string(), nullable=False),
            pa.field("label", pa.string()),
            pa.field("source_page", pa.string()),
        ]
    )


def claims_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("page_id", pa.string(), nullable=False),
            pa.field("claim_text", pa.string()),
            pa.field("value_type", pa.string()),
            pa.field("value", pa.string()),
            pa.field("unit", pa.string()),
            pa.field("confidence", pa.float64()),
            pa.field("confidence_label", pa.string()),
            pa.field("source_ref", pa.string()),
        ]
    )


def sources_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("location_uri", pa.string(), nullable=False),
            pa.field("location_kind", pa.string(), nullable=False),
            pa.field("source_content_hash", pa.string()),
            pa.field("fetched_at", pa.timestamp("us", tz="UTC")),
            pa.field("last_verified_at", pa.timestamp("us", tz="UTC")),
            pa.field("structure_ref", pa.string()),
            pa.field("ignored_json", pa.string()),  # JSON-encoded list of ignored filenames
            # reserved veracity / quality fields (Phase 3)
            pa.field("quality_score", pa.float64()),
            pa.field("veracity_score", pa.float64()),
            pa.field("evaluated_at", pa.timestamp("us", tz="UTC")),
            pa.field("evaluation_notes", pa.string()),
        ]
    )


# ---- connection / table creation ----


def connect(smalt_root: Path) -> lancedb.DBConnection:
    """Open (or create) the LanceDB connection for this Smalt."""
    import lancedb

    target = lance_dir(smalt_root)
    target.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(target))


def _existing_table_names(db: lancedb.DBConnection) -> list[str]:
    """Return the names of existing tables.

    LanceDB >= 0.30 returns a `ListTablesResponse` with a `.tables`
    attribute (a list of table names). The minimum version is pinned in
    pyproject.toml, so the older list-of-strings return shape isn't
    handled here anymore.
    """
    return [str(t) for t in db.list_tables().tables]


def ensure_tables(smalt_root: Path, *, embedding_dim: int) -> list[str]:
    """Create any missing tables. Returns the list of tables that were created.

    Idempotent: existing tables are left untouched.
    """
    db = connect(smalt_root)
    existing = set(_existing_table_names(db))
    schemas: dict[str, pa.Schema] = {
        TABLE_PAGES: pages_schema(),
        TABLE_EMBEDDINGS: embeddings_schema(embedding_dim),
        TABLE_LINKS: links_schema(),
        TABLE_CLAIMS: claims_schema(),
        TABLE_SOURCES: sources_schema(),
    }
    created: list[str] = []
    for name, schema in schemas.items():
        if name in existing:
            continue
        db.create_table(name, schema=schema, mode="create")
        created.append(name)
    return created


def list_tables(smalt_root: Path) -> list[str]:
    return _existing_table_names(connect(smalt_root))


# ---- upsert + delete helpers ----


def _open_table(db: lancedb.DBConnection, name: str) -> Any:
    return db.open_table(name)


def sql_str(value: str) -> str:
    """Escape a string for safe inclusion in a LanceDB / SQL filter expression.

    LanceDB's `delete(filter)` takes a SQL expression string with no
    parameter-binding support, so untrusted values (page ids, source ids,
    file paths) interpolated into the filter must have single quotes
    doubled per the SQL string-literal rule. Without this, a frontmatter
    `id: "x' OR '1'='1"` would delete every row in the table.
    """
    return "'" + value.replace("'", "''") + "'"


def upsert_pages(db: lancedb.DBConnection, rows: list[dict[str, Any]]) -> None:
    """Upsert page rows by `id`. Rows must match the schema produced by `pages_schema()`."""
    if not rows:
        return
    table = _open_table(db, TABLE_PAGES)
    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)


def upsert_embeddings(db: lancedb.DBConnection, rows: list[dict[str, Any]]) -> None:
    """Upsert embedding rows by `page_id`."""
    if not rows:
        return
    table = _open_table(db, TABLE_EMBEDDINGS)
    table.merge_insert("page_id").when_matched_update_all().when_not_matched_insert_all().execute(
        rows
    )


def replace_links_for_page(
    db: lancedb.DBConnection, page_id: str, rows: list[dict[str, Any]]
) -> None:
    """Replace all outgoing links from a single page.

    Links are page-owned; when a page is re-indexed we drop its old links
    and write the new ones. This avoids a fragile per-edge upsert.
    """
    table = _open_table(db, TABLE_LINKS)
    table.delete(f"from_id = {sql_str(page_id)}")
    if rows:
        table.add(rows)


def replace_claims_for_page(
    db: lancedb.DBConnection, page_id: str, rows: list[dict[str, Any]]
) -> None:
    """Replace all claims attached to a single page (same rationale as links)."""
    table = _open_table(db, TABLE_CLAIMS)
    table.delete(f"page_id = {sql_str(page_id)}")
    if rows:
        table.add(rows)


def existing_page_hashes(db: lancedb.DBConnection) -> dict[str, str]:
    """Return `{page_id: content_hash}` for every page already in the index.

    Used by the indexer to skip files whose content hasn't changed.

    Performance note: this scans the whole pages table, but projects only
    the two columns it needs (`id`, `content_hash`) so memory and bandwidth
    don't scale with how big the body / frontmatter columns get.
    """
    table = _open_table(db, TABLE_PAGES)
    if table.count_rows() == 0:
        return {}
    arrow = table.to_arrow().select(["id", "content_hash"])
    ids = arrow.column("id").to_pylist()
    hashes = arrow.column("content_hash").to_pylist()
    return {str(i): str(h) for i, h in zip(ids, hashes, strict=True) if i is not None}


def fetch_created_at(db: lancedb.DBConnection, page_ids: set[str]) -> dict[str, Any]:
    """Look up `created_at` for the given page ids, projecting only what's needed.

    Returns a `{id: created_at}` dict; ids that don't exist (or whose
    created_at is null) are omitted. Used by the indexer so updates
    preserve the original creation timestamp rather than resetting it.
    """
    if not page_ids:
        return {}
    table = _open_table(db, TABLE_PAGES)
    if table.count_rows() == 0:
        return {}
    quoted_ids = ", ".join(sql_str(pid) for pid in page_ids)
    arrow = (
        table.search()
        .where(f"id IN ({quoted_ids})")
        .select(["id", "created_at"])
        .limit(len(page_ids))
        .to_arrow()
    )
    out: dict[str, Any] = {}
    ids = arrow.column("id").to_pylist()
    timestamps = arrow.column("created_at").to_pylist()
    for i, ts in zip(ids, timestamps, strict=True):
        if i is not None and ts is not None:
            out[str(i)] = ts
    return out


# ---- index management ----


def create_or_refresh_fts(db: lancedb.DBConnection, *, replace: bool = True) -> None:
    """(Re)create the full-text-search indexes — one per field.

    LanceDB's native FTS only accepts one field per index, so we create
    separate indexes on `title` and `body`. Hybrid search handles them at
    query time.
    """
    table = _open_table(db, TABLE_PAGES)
    for field in ("title", "body"):
        with contextlib.suppress(Exception):  # pragma: no cover — best effort
            table.create_fts_index(field, replace=replace)


def create_or_refresh_vector_index(
    db: lancedb.DBConnection, *, num_partitions: int | None = None, replace: bool = True
) -> None:
    """(Re)create the ANN index on `embeddings.vector`.

    Falls back gracefully on tables too small to index (LanceDB requires a
    minimum row count). At small Smalt sizes we just rely on a brute-force
    scan, which is fine.
    """
    table = _open_table(db, TABLE_EMBEDDINGS)
    if table.count_rows() < 256:
        return
    kwargs: dict[str, Any] = {"replace": replace, "metric": "cosine"}
    if num_partitions is not None:
        kwargs["num_partitions"] = num_partitions
    with contextlib.suppress(Exception):  # pragma: no cover — best-effort; brute force still works
        table.create_index(**kwargs)
