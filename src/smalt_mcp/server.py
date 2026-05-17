"""FastAPI app construction + MCP server wiring + lifespan.

Mounts a Streamable-HTTP MCP transport at `/sse` (same pattern as deco-assaying).
Tools are defined in `smalt_mcp.tools` — this module only does the plumbing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel
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


# `/admin/health` returns the detailed observability payload assembled
# by `App.index_status_payload()`. Same payload as the `index_status` MCP
# tool — use whichever channel is more convenient.
#
# Distinct from `/health` (deliberately minimal load-balancer probe):
# this endpoint is for operators + monitoring, the other is for "is the
# process up?" checks. The shape is intentionally not pinned to a
# `response_model` Pydantic class — the payload is rich enough that
# pinning would force a parallel-maintained schema, and the only
# downside is no automatic OpenAPI schema gen for the body.
@router.get("/admin/health", tags=["admin"])
async def admin_health() -> dict[str, Any]:
    return _app_instance.index_status_payload()


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
