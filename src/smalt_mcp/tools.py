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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp import types

from smalt_mcp.permissions import Scope
from smalt_mcp.storage import lance

if TYPE_CHECKING:
    from smalt_mcp.app import App


# ---- ToolDef + handler signature ----


Handler = Callable[["App", dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDef:
    """One MCP tool: spec + handler + permission scope."""

    spec: types.Tool
    scope: Scope
    handler: Handler


# ---- handlers ----


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

    # Smalt dir exists — try to open the LanceDB connection. If the index
    # subdirectory hasn't been created yet, tables list is empty.
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
