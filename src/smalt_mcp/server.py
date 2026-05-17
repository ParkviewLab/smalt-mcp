"""FastAPI app construction + MCP server wiring + lifespan.

Mounts a Streamable-HTTP MCP transport at `/sse` (same pattern as deco-assaying).
Tools are defined in `smalt_mcp.tools` — this module only does the plumbing.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tarfile
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from starlette.routing import Route

from smalt_mcp import tools as tools_module
from smalt_mcp.app import App
from smalt_mcp.config import VERSION
from smalt_mcp.permissions import Scope

logger = logging.getLogger(__name__)
_started_at = time.time()


# ---------------------------------------------------------------------------
# Shared App instance + scope
#
# We construct the `App` at module-import time so it's available to both the
# MCP tool handlers and the FastAPI routes. Heavy resources (embedder, db
# connection) inside `App` are lazy — first use loads them; importing this
# module doesn't.

_app_instance = App()


def _server_scope() -> Scope:
    """Read the server-wide scope at startup.

    Accepted values (with tier):
      - `read_only`           (0): only `Scope.READ_ONLY` tools exposed.
      - `read_write`          (1, default): READ_ONLY + READ_WRITE.
      - `remove_destructive`  (2): READ_ONLY + READ_WRITE + REMOVE_DESTRUCTIVE.

    Default is `read_write` so the destructive tools (`remove_page`,
    `update_claim`, `remove_claim`, `remove_link`) are opt-in to expose —
    they're powerful enough that operators should consciously enable them.
    """
    raw = (os.environ.get("SMALT_SCOPE") or "read_write").lower()
    if raw == "read_only":
        return Scope.READ_ONLY
    if raw == "read_write":
        return Scope.READ_WRITE
    if raw == "remove_destructive":
        return Scope.REMOVE_DESTRUCTIVE
    raise ValueError(
        f"invalid SMALT_SCOPE={raw!r}; expected one of read_only / read_write / remove_destructive"
    )


_SERVER_SCOPE: Scope = _server_scope()


# ---------------------------------------------------------------------------
# MCP server (mounted at /sse via the FastAPI app below)

mcp = Server("smalt-mcp", version=VERSION)
session_manager = StreamableHTTPSessionManager(app=mcp, stateless=True)


@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    return tools_module.list_tools(_SERVER_SCOPE)


@mcp.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        result = await tools_module.dispatch(
            name, arguments, app=_app_instance, scope=_SERVER_SCOPE
        )
    except KeyError as e:
        return _ok({"error": "unknown_tool", "message": str(e)})
    except PermissionError as e:
        return _ok({"error": "forbidden", "message": str(e)})
    except Exception as e:  # noqa: BLE001 — surface the error to the LLM as a tool_result
        logger.exception("tool %s raised", name)
        return _ok({"error": "tool_error", "message": str(e), "type": type(e).__name__})
    return _ok(result)


def _ok(payload: dict[str, Any]) -> list[types.TextContent]:
    """Wrap a JSON-serializable payload as a single MCP text content block."""
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]


# ---------------------------------------------------------------------------
# /sse — Streamable HTTP MCP transport mounted as raw ASGI3.
# (Class instance, not bare async function, so Starlette doesn't wrap it.)


class MCPASGIApp:
    async def __call__(self, scope, receive, send) -> None:
        await session_manager.handle_request(scope, receive, send)


mcp_asgi = MCPASGIApp()


# ---------------------------------------------------------------------------
# Lifespan — runs the MCP session manager.


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    uvlog = logging.getLogger("uvicorn.error")
    async with session_manager.run():
        uvlog.info(
            "smalt-mcp v%s ready (scope=%s, smalt_dir=%s)",
            VERSION,
            _SERVER_SCOPE.value,
            _app_instance.cfg.smalt_dir,
        )
        yield


# ---------------------------------------------------------------------------
# /health + /admin/version (read-only ops endpoints, like deco-assaying's)


router = APIRouter()


class Health(BaseModel):
    ok: bool
    version: str
    uptime_seconds: float


@router.get("/health", response_model=Health, tags=["health"])
async def health() -> Health:
    return Health(ok=True, version=VERSION, uptime_seconds=time.time() - _started_at)


class AdminVersion(BaseModel):
    name: str
    version: str
    scope: str
    smalt_dir: str
    smalt_exists: bool


@router.get("/admin/version", response_model=AdminVersion, tags=["admin"])
async def admin_version() -> AdminVersion:
    return AdminVersion(
        name="smalt-mcp",
        version=VERSION,
        scope=_SERVER_SCOPE.value,
        smalt_dir=str(_app_instance.cfg.smalt_dir),
        smalt_exists=_app_instance.smalt_exists(),
    )


# ---------------------------------------------------------------------------
# /admin/backup — streaming tar.gz of the Smalt directory.
#
# Design recap (full rationale in the Workstream C plan):
#
# - **Streaming, not in-memory + not disk-tmp**: a Smalt can grow to GB
#   scale (pages + LanceDB lance files + sidecar JSON structures);
#   in-memory zip risks OOM; disk-tmp pays 2× I/O + tmp-space hygiene.
#   `tarfile.open(fileobj=stream, mode="w|gz")` is built for streaming
#   ("|gz" = pipe-mode gzip; no seek required, constant memory buffer).
#
# - **tar.gz, not zip**: zip's central-directory-at-end of file requires
#   seek() back to rewrite offsets, breaking pure streaming. tar.gz is
#   stdlib + naturally streaming. If Windows-double-click compatibility
#   becomes a hard requirement later, swap to `zipstream-ng` and add a
#   `?format=zip` branch — endpoint shape stays the same.
#
# - **Best-effort consistency, not strict**: enumerate file list inside
#   the corpus mutex (instantaneous); stream bytes outside it (slow;
#   would block writers for minutes on a multi-GB backup). Each tar
#   member is a snapshot of the file at the moment tarfile.addfile()
#   reads it. Atomic tmp-then-rename writes (already in place across
#   every write path) prevent torn writes WITHIN a single file —
#   concurrent writers' bytes are either fully old or fully new in the
#   archive. The remaining anomaly: file A might be at version N and
#   file B at version N+1 in the same backup if a writer commits
#   mid-stream. Acceptable for a substrate-level backup; transactional
#   snapshots are LanceDB time-travel territory, out of scope.

_VALID_BACKUP_FORMATS: frozenset[str] = frozenset({"tar.gz"})
"""Wire-format whitelist. v1 only supports tar.gz; `?format=zip` is
reserved for a future PR that pulls in zipstream-ng."""

_VALID_BACKUP_CONTENTS: frozenset[str] = frozenset({"full", "no-index", "pages-only"})
"""Scope-filter whitelist for the backup file enumeration."""


def _enumerate_backup_files(smalt_dir: Path, contents: str) -> list[Path]:
    """Walk `smalt_dir` and return absolute paths of every file in scope.

    Filtered by `contents`:
      - `full` — every file under `smalt_dir`.
      - `no-index` — every file EXCEPT under `smalt_dir/index/lance/`
        (the LanceDB store is rebuildable from `pages/`; skipping it
        cuts archive size roughly in half).
      - `pages-only` — only files under `pages/`, `structures/`, and
        `schema/`. Smallest possible backup; restore requires a full
        bootstrap + indexer rebuild.

    Returns paths sorted for deterministic archive ordering.
    """
    if not smalt_dir.exists():
        return []

    if contents == "full":
        keep = lambda rel: True  # noqa: E731
    elif contents == "no-index":
        keep = lambda rel: not rel.startswith("index/lance/") and rel != "index/lance"  # noqa: E731
    elif contents == "pages-only":
        keep = lambda rel: (  # noqa: E731
            rel.startswith("pages/")
            or rel.startswith("structures/")
            or rel.startswith("schema/")
        )
    else:
        raise ValueError(f"unknown contents: {contents!r}")

    out: list[Path] = []
    for p in smalt_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = str(p.relative_to(smalt_dir))
        except ValueError:
            continue  # path escapes the root via symlink; skip
        if keep(rel):
            out.append(p)
    out.sort()
    return out


def _stream_tar_gz(files: list[Path], root: Path) -> Iterator[bytes]:
    """Generator: yields gzip-compressed tar bytes, one file at a time.

    Uses an in-memory `io.BytesIO()` as the tar's file-object backing.
    After each `tar.add(path)`, drains the buffer and yields the bytes
    — memory stays bounded to "one file's tar entry overhead" rather
    than growing to the size of the whole archive.

    Caveat for very large individual files (a multi-GB lance file
    would be unusual but possible): `tar.add()` reads the whole file
    into the buffer before yield resumes. For now we accept that; if
    a real-world case hits it, swap to manual `tarfile.TarInfo` +
    chunked reads. Defer the complexity until a real workload needs it.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w|gz") as tar:
        for path in files:
            try:
                arcname = str(path.relative_to(root))
            except ValueError:
                continue
            try:
                tar.add(path, arcname=arcname, recursive=False)
            except (OSError, PermissionError) as e:
                # A file disappeared between enumeration and read, or we
                # don't have permission. Log + skip — best-effort.
                logger.warning("backup: skipping %s (%s)", arcname, e)
                continue
            chunk = buf.getvalue()
            if chunk:
                buf.seek(0)
                buf.truncate(0)
                yield chunk
    # Flush the gzip trailer (written when the tarfile context manager
    # exits — see the `with` block above).
    final = buf.getvalue()
    if final:
        yield final


@router.get("/admin/backup", tags=["admin"])
async def admin_backup(
    format: str = Query("tar.gz", description="Archive format. v1: tar.gz only."),
    contents: str = Query(
        "full",
        description="Scope filter: full | no-index | pages-only.",
    ),
) -> StreamingResponse:
    """Stream a tar.gz of the Smalt directory.

    Query params:
      - `format=tar.gz` (default; only supported value in v1).
      - `contents=full|no-index|pages-only` (default `full`).

    Response: `Content-Type: application/gzip`, `Content-Disposition:
    attachment; filename="smalt-<dirname>-<utc-iso>.tar.gz"`. No
    `Content-Length` (chunked transfer; size unknown until done).

    Consistency: best-effort — file list enumerated under the corpus
    mutex; bytes streamed outside the mutex. Per-file atomicity is
    preserved by the existing tmp-then-rename write discipline; across
    files a writer committing mid-stream may produce a "version-N
    fileA + version-(N+1) fileB" archive. Acceptable for backup;
    transactional snapshots are out of scope.
    """
    if format not in _VALID_BACKUP_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"format must be one of {sorted(_VALID_BACKUP_FORMATS)}; "
                f"got {format!r}. (zipstream-ng support is reserved for a "
                "future PR.)"
            ),
        )
    if contents not in _VALID_BACKUP_CONTENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"contents must be one of {sorted(_VALID_BACKUP_CONTENTS)}; "
                f"got {contents!r}."
            ),
        )

    smalt_dir = _app_instance.cfg.smalt_dir

    # Phase 1 (under mutex, instantaneous): snapshot the file list.
    # Phase 2 (outside mutex, slow): stream the bytes. The streamer
    # yields per-file chunks; writers concurrent with the stream see no
    # intermediate state because each individual file's write is atomic
    # (tmp-then-rename), so the streamer reads either the pre-commit or
    # post-commit bytes of each file — never half-and-half.
    with _app_instance.mutex.acquire("backup_enumerate"):
        files = _enumerate_backup_files(smalt_dir, contents)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"smalt-{smalt_dir.name}-{ts}.tar.gz"
    return StreamingResponse(
        _stream_tar_gz(files, smalt_dir),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # `Content-Encoding: identity` is the explicit "this body is
            # already in its final form" signal — keeps the FastAPI app's
            # GZipMiddleware (mounted further down with minimum_size=256)
            # from double-gzipping our already-gzipped tar stream. Without
            # this header, the stream would be re-compressed AND
            # transparently decompressed by the client, leaving a plain
            # tar file on disk (named ".tar.gz" but not actually gzipped).
            "Content-Encoding": "identity",
            # Cache-control hint for HTTP caches that might sit in front.
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# FastAPI app


app = FastAPI(
    title="smalt-mcp",
    version=VERSION,
    description=(
        "MCP server wrapping the Smalt's storage surface (read/write/link/claim/"
        "search) for ParkviewLab's CoGrind project. /admin/* endpoints expose "
        "read-only ops information; tool calls go to /sse over MCP."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=256)
# Streamable HTTP MCP transport at /sse, mounted as a raw ASGI3 endpoint so
# Starlette doesn't wrap it in request_response (which would break SSE
# streaming semantics).
app.router.routes.append(Route("/sse", endpoint=mcp_asgi, methods=["GET", "POST", "DELETE"]))
app.include_router(router)
