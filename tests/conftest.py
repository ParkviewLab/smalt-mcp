"""Shared test fixtures.

The MCP `/sse` transport relies on a `StreamableHTTPSessionManager` that
hard-errors on `run()` being called twice. Each `with TestClient(app) as c`
block runs the FastAPI lifespan, which calls `session_manager.run()`. So we
can't have two MCP-using test modules each owning their own `with`-scoped
client — the second module's lifespan startup blows up.

The fix: a single session-scoped TestClient lives here; every test module
that touches `/sse` reuses it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def mcp_client(tmp_path_factory) -> TestClient:
    """One TestClient for the whole test session — drives the MCP `/sse`
    surface, lifespan started exactly once.

    Points SMALT_DIR at a per-session temp directory so tests don't touch
    the user's real Smalt.
    """
    import os

    # Server module reads SMALT_DIR at import time via load_config(); set the
    # env var before the import path triggers.
    smalt_dir = tmp_path_factory.mktemp("smalt")
    os.environ["SMALT_DIR"] = str(smalt_dir)

    from smalt_mcp.server import app

    with TestClient(app) as c:
        yield c
