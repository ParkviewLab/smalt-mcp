"""Concurrency assertions for the MCP server (Workstream E).

Separated from `test_server.py` because it uses a different test
client pattern: `httpx.AsyncClient(transport=ASGITransport(app=...))`
+ `asyncio.gather()` to fire genuinely concurrent requests against
the FastAPI ASGI app. The existing `test_server.py` uses
`starlette.testclient.TestClient` (sync) which cannot express
"these N requests are in-flight at the same time."

Tests verify:
- E-2: N concurrent reads run in parallel (wall-clock proves the
  event loop isn't blocked by the work).
- E-2: a slow read does NOT block other handlers — the event loop
  stays free to dispatch.
- E-3 (later PR): writes serialize at the corpus mutex; reads
  overlap with in-flight writes; mutex contention counters move.

All tests run against the same long-lived test app instance from
the session-scoped fixture; tests must be tolerant of accumulated
state from earlier tests in the same session (the seed Smalt + any
pages added by tests that happened to run first).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


# ---- session-scoped ASGI app + smalt setup ----
#
# Constructed once per session, matching the pattern in
# tests/conftest.py but using a separate smalt_dir so the
# concurrency tests don't pollute the test_server.py seed (and
# vice-versa). The fastembed model load is expensive; the eager
# init in lifespan handles it once at startup.


@pytest.fixture(scope="module")
def _concurrency_smalt_dir(tmp_path_factory) -> Path:
    smalt_dir = tmp_path_factory.mktemp("concurrency_smalt")
    return smalt_dir


@pytest.fixture  # function-scoped — see _concurrency_app docstring
def _concurrency_app(_concurrency_smalt_dir):
    """Build a fresh ASGI app pointing at a dedicated smalt_dir, with
    the seed pages indexed so reads return results.

    Function-scoped because `StreamableHTTPSessionManager` is a
    module-level singleton in `smalt_mcp.server` that can only be
    `run()` once per instance. To start the lifespan cleanly per
    test, we reload `smalt_mcp.server` so a fresh session_manager
    gets constructed. The smalt_dir (bootstrap + indexed seed) is
    reused via `_concurrency_smalt_dir` (module-scoped) — only the
    server module gets recreated.
    """
    # Set env BEFORE the server module imports its Config.
    os.environ["SMALT_DIR"] = str(_concurrency_smalt_dir)
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    os.environ["EMBEDDING_DIM"] = "384"
    os.environ["SMALT_SCOPE"] = "remove_destructive"

    # Seed the same shape as conftest._SEED_PAGES so search returns
    # something. Copy the conftest helper inline (small).
    _seed = {
        "pages/entities/alice.md": """---
id: ent-alice
type: entity
title: Alice
aliases: [Alicia]
entity_kind: person
---
Alice is a fictional person used in the concurrency-test seed Smalt.
""",
        "pages/entities/bob.md": """---
id: ent-bob
type: entity
title: Bob
entity_kind: person
---
Bob is another fictional person in the concurrency-test seed Smalt.
""",
        "pages/concepts/cs.md": """---
id: con-cs
type: concept
title: Computer Science
is_domain: true
---
Computer Science is a seed domain ConceptPage.
""",
    }
    for rel, content in _seed.items():
        target = _concurrency_smalt_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    # Bootstrap + index.
    from smalt_mcp.config import load_config
    from smalt_mcp.storage.embedder import make_embedder
    from smalt_mcp.storage.indexer import Indexer
    from smalt_mcp.storage.lance import connect, ensure_tables

    cfg = load_config()
    ensure_tables(_concurrency_smalt_dir, embedding_dim=cfg.embedding.dim)
    embedder = make_embedder(cfg)
    db = connect(_concurrency_smalt_dir)
    Indexer(smalt_root=_concurrency_smalt_dir, embedder=embedder, db=db).run()

    # Import the server AFTER env is set so config.load_config picks it up.
    # Force a re-import in case a previous test module already imported it
    # against a different smalt_dir.
    import importlib
    import smalt_mcp.server as server_mod

    importlib.reload(server_mod)
    return server_mod.app


@pytest_asyncio.fixture
async def async_client(_concurrency_app) -> AsyncIterator[AsyncClient]:
    """An httpx AsyncClient bound to the ASGI app via ASGITransport,
    with the FastAPI lifespan explicitly driven by LifespanManager.

    httpx's ASGITransport does NOT trigger lifespan events; we need
    `asgi_lifespan.LifespanManager` to start the StreamableHTTPSessionManager's
    task group + the C-13 scheduler + the E-1 eager init.

    Function-scoped: the StreamableHTTPSessionManager's session state
    leaks between tests in module scope (a test 1 session ID stays
    "in-flight" enough to deadlock test 2's concurrent requests).
    Function scope means each test starts the lifespan fresh — costs
    ~1s of fastembed model load per test (with fake embedder),
    acceptable for the handful of E-2 + E-3 concurrency tests.
    """
    async with LifespanManager(_concurrency_app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# ---- helpers ----


@contextlib.contextmanager
def _swap_tool_handler(
    tool_name: str, new_sync_fn: Callable[..., dict[str, Any]]
):
    """Temporarily replace a tool's handler with one wrapping `new_sync_fn`
    via `_wrap_sync_in_thread`. Restores the original on exit.

    The ToolDef is a frozen dataclass; we use `object.__setattr__` to
    bypass the freeze (and restore the original the same way). This
    is the only safe pattern for monkeypatching tool handlers
    cross-test — monkeypatch.setattr can't write to frozen dataclass
    attrs, and replacing the module-level `tools.<name>` doesn't
    affect dispatch (which goes through `_TOOLS_BY_NAME`).
    """
    from smalt_mcp import tools

    td = tools._TOOLS_BY_NAME[tool_name]
    original = td.handler
    object.__setattr__(td, "handler", tools._wrap_sync_in_thread(new_sync_fn))
    try:
        yield
    finally:
        object.__setattr__(td, "handler", original)


async def _mcp_init(client: AsyncClient) -> str:
    """Initialize an MCP session; return the session id."""
    r = await client.post(
        "/sse",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "concurrency-test", "version": "0.0"},
            },
        },
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 200, r.text
    return r.headers.get("mcp-session-id", "")


async def _call_tool(
    client: AsyncClient,
    session_id: str,
    name: str,
    arguments: dict,
    *,
    req_id: int = 100,
) -> dict:
    """Fire a tools/call MCP request; return the parsed result payload."""
    r = await client.post(
        "/sse",
        json={
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id,
        },
    )
    assert r.status_code == 200, r.text
    # Streamable HTTP can return either application/json or SSE
    # data: <json> lines. Handle both.
    ct = r.headers.get("content-type", "")
    if ct.startswith("application/json"):
        body = r.json()
    else:
        # SSE: last data: line is the response.
        msgs = []
        for line in r.text.splitlines():
            if line.startswith("data: "):
                msgs.append(json.loads(line[6:]))
            elif line.startswith("data:"):
                msgs.append(json.loads(line[5:]))
        assert msgs, f"no SSE data lines in: {r.text!r}"
        body = msgs[-1]
    assert "result" in body, f"tools/call returned: {body!r}"
    contents = body["result"]["content"]
    assert contents and contents[0]["type"] == "text"
    return json.loads(contents[0]["text"])


# ---- E-2 tests ----


@pytest.mark.asyncio
async def test_e2_concurrent_reads_run_in_parallel(async_client):
    """Three concurrent `search` MCP requests complete in wall-clock
    time substantially less than 3× single-call latency.

    Sync function sleeps 0.3s; sequential = 0.9s, parallel ≈ 0.3s.
    Bound: wall-clock < 0.9s × 0.7 = 0.63s (parallelism factor ≥ ~1.4×).
    Real-world parallelism is much higher (~0.3s actual) but the
    ceiling gives slack for thread-pool warmup + httpx + MCP overhead.
    """
    canned = {"results": [], "count": 0, "_concurrency_test": True}
    call_count = 0

    def _slow_search(app, arguments):
        nonlocal call_count
        call_count += 1
        time.sleep(0.3)
        return canned

    with _swap_tool_handler("search", _slow_search):
        sid = await _mcp_init(async_client)
        started = time.perf_counter()
        results = await asyncio.gather(
            _call_tool(async_client, sid, "search", {"query": "x", "top_k": 1}, req_id=201),
            _call_tool(async_client, sid, "search", {"query": "y", "top_k": 1}, req_id=202),
            _call_tool(async_client, sid, "search", {"query": "z", "top_k": 1}, req_id=203),
        )
        elapsed = time.perf_counter() - started

    assert call_count == 3, f"expected 3 calls, got {call_count}"
    assert all(r.get("_concurrency_test") is True for r in results)
    assert elapsed < 0.9 * 0.7, (
        f"3 concurrent searches took {elapsed:.3f}s wall-clock; "
        f"expected <{0.9 * 0.7:.3f}s (parallelism factor ≥ 1.4×). "
        "Event loop is likely blocked on sync work — to_thread wrap missing."
    )


@pytest.mark.asyncio
async def test_e2_event_loop_not_blocked_during_slow_read(async_client):
    """A slow `search` does NOT block a concurrent `status` call —
    the event loop stays free to dispatch other handlers.

    Without the to_thread wrap, the slow search would freeze the
    loop and status would be queued behind it. With the wrap, both
    requests run concurrently — status returns in <100ms while
    search is still sleeping.
    """
    def _slow_search(app, arguments):
        time.sleep(0.5)
        return {"results": [], "count": 0}

    with _swap_tool_handler("search", _slow_search):
        sid = await _mcp_init(async_client)
        # Fire both in parallel via gather. The slow one is search;
        # status is fast. asyncio.gather collects both. If the loop
        # is blocked, both serialize and total elapsed is ~0.5s.
        # If not, both run concurrently and total is also ~0.5s
        # (bounded by the slow one). The discriminating signal is
        # the INDIVIDUAL status latency, not total.
        async def _timed_status():
            start = time.perf_counter()
            r = await _call_tool(async_client, sid, "status", {}, req_id=222)
            return r, time.perf_counter() - start

        async def _timed_search():
            start = time.perf_counter()
            r = await _call_tool(async_client, sid, "search", {"query": "x"}, req_id=221)
            return r, time.perf_counter() - start

        (search_result, search_latency), (status_result, status_latency) = await asyncio.gather(
            _timed_search(), _timed_status()
        )

    # The status call's own latency must be much less than the
    # search's (which is bounded below by the 0.5s sleep). If the
    # event loop is blocked, status would be queued and its latency
    # would be near search's.
    assert status_latency < 0.3, (
        f"status latency {status_latency:.3f}s suggests the event loop is "
        f"blocked by the in-flight search (which took {search_latency:.3f}s). "
        "to_thread wrap on `status` or `search` is missing."
    )
    assert search_latency >= 0.5, (
        f"search returned in {search_latency:.3f}s — slower-search monkeypatch didn't take effect"
    )
    assert "exists" in status_result or "smalt_dir" in status_result
    assert search_result.get("count") == 0


@pytest.mark.asyncio
async def test_e2_thread_pool_workers_config_respected(monkeypatch):
    """`SMALT_THREAD_POOL_WORKERS` env var changes the configured worker count."""
    from smalt_mcp.config import load_config

    monkeypatch.setenv("SMALT_THREAD_POOL_WORKERS", "8")
    cfg = load_config()
    assert cfg.thread_pool_workers == 8

    monkeypatch.setenv("SMALT_THREAD_POOL_WORKERS", "64")
    cfg = load_config()
    assert cfg.thread_pool_workers == 64

    # Invalid (zero / negative) falls back to default 32.
    monkeypatch.setenv("SMALT_THREAD_POOL_WORKERS", "0")
    cfg = load_config()
    assert cfg.thread_pool_workers == 32

    monkeypatch.setenv("SMALT_THREAD_POOL_WORKERS", "-5")
    cfg = load_config()
    assert cfg.thread_pool_workers == 32


@pytest.mark.asyncio
async def test_e2_thread_pool_default_when_unset(monkeypatch):
    """Without the env var, default is 32."""
    from smalt_mcp.config import load_config

    monkeypatch.delenv("SMALT_THREAD_POOL_WORKERS", raising=False)
    cfg = load_config()
    assert cfg.thread_pool_workers == 32


# ---- E-3 tests: writes serialize at the mutex; reads overlap writes ----


@pytest.mark.asyncio
async def test_e3_writes_serialize_at_mutex(async_client):
    """Two concurrent `write_page` requests with a slow indexer step
    MUST run sequentially — the corpus mutex (single-writer) enforces
    that. We swap `write_page` with a slow stub that records
    start/end timestamps in shared state, fire two in parallel via
    gather, and assert the intervals don't overlap.

    This proves: (a) the to_thread wrap let both writes get scheduled
    immediately, (b) the threading.Lock corpus mutex serialized them
    correctly inside the threads.
    """
    intervals: list[tuple[str, float, float]] = []
    intervals_lock = __import__("threading").Lock()

    def _slow_write_page(app, arguments):
        # Acquire the corpus mutex like the real write_page does —
        # without this, the test would prove nothing about mutex
        # behavior, only about the thread pool.
        with app.mutex.acquire("e3_test_write_page"):
            tag = arguments.get("frontmatter", {}).get("id", "?")
            start = time.perf_counter()
            time.sleep(0.2)
            end = time.perf_counter()
            with intervals_lock:
                intervals.append((tag, start, end))
        return {"id": tag, "_test": True}

    with _swap_tool_handler("write_page", _slow_write_page):
        sid = await _mcp_init(async_client)
        results = await asyncio.gather(
            _call_tool(
                async_client, sid, "write_page",
                {"frontmatter": {"id": "ent-e3-a", "type": "entity", "title": "A", "entity_kind": "test"}},
                req_id=301,
            ),
            _call_tool(
                async_client, sid, "write_page",
                {"frontmatter": {"id": "ent-e3-b", "type": "entity", "title": "B", "entity_kind": "test"}},
                req_id=302,
            ),
        )

    assert len(intervals) == 2, f"expected 2 intervals, got {len(intervals)}"
    # Sort by start time. The two intervals must be strictly
    # non-overlapping (second.start >= first.end).
    intervals.sort(key=lambda t: t[1])
    first_tag, first_start, first_end = intervals[0]
    second_tag, second_start, second_end = intervals[1]
    assert second_start >= first_end, (
        f"writes overlapped! first {first_tag}:[{first_start:.3f},{first_end:.3f}] "
        f"second {second_tag}:[{second_start:.3f},{second_end:.3f}] — mutex didn't serialize"
    )
    for r in results:
        assert r.get("_test") is True


@pytest.mark.asyncio
async def test_e3_read_overlaps_with_in_flight_write(async_client):
    """While a slow `write_page` holds the corpus mutex, a concurrent
    `search` read MUST complete (reads don't acquire the mutex).
    Proves the event loop isn't blocked by the write thread."""
    write_started = __import__("threading").Event()

    def _slow_write_page(app, arguments):
        with app.mutex.acquire("e3_test_overlap_write"):
            write_started.set()
            time.sleep(0.5)
        return {"id": "ent-e3-overlap", "_test": True}

    def _fast_search(app, arguments):
        return {"results": [], "count": 0, "_overlap_test": True}

    with _swap_tool_handler("write_page", _slow_write_page), \
         _swap_tool_handler("search", _fast_search):
        sid = await _mcp_init(async_client)
        # Start the write; wait for it to actually acquire the mutex
        # (so we know the search runs DURING the write, not before).
        write_task = asyncio.create_task(
            _call_tool(
                async_client, sid, "write_page",
                {"frontmatter": {"id": "ent-e3-overlap", "type": "entity", "title": "x", "entity_kind": "test"}},
                req_id=311,
            )
        )
        # Poll for write_started flag (set inside the mutex).
        for _ in range(50):
            if write_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert write_started.is_set(), "write didn't enter the mutex in time"

        # Now fire the search. It should return quickly even though
        # the write is still holding the mutex (reads don't acquire it).
        search_start = time.perf_counter()
        search_result = await _call_tool(async_client, sid, "search", {"query": "x"}, req_id=312)
        search_latency = time.perf_counter() - search_start

        # Wait for the write to finish so we don't leak the task.
        write_result = await write_task

    assert search_result.get("_overlap_test") is True
    assert search_latency < 0.3, (
        f"search took {search_latency:.3f}s during in-flight write — "
        "loop was blocked or reads are serializing on the mutex (regression)"
    )
    assert write_result.get("_test") is True


@pytest.mark.asyncio
async def test_e3_mutex_contention_counters_advance(async_client):
    """After two concurrent writes serialize on the mutex,
    `index_status.mutex.acquire_count` advances by at least 2 and
    `mean_wait_ms` is > 0 (the second write actually waited)."""
    def _slow_write_page(app, arguments):
        with app.mutex.acquire("e3_test_contention"):
            time.sleep(0.15)
        return {"id": arguments.get("frontmatter", {}).get("id", "?"), "_test": True}

    with _swap_tool_handler("write_page", _slow_write_page):
        sid = await _mcp_init(async_client)

        # Read baseline contention counters.
        before = await _call_tool(async_client, sid, "index_status", {}, req_id=320)
        baseline_count = before["mutex"]["acquire_count"]

        await asyncio.gather(
            _call_tool(
                async_client, sid, "write_page",
                {"frontmatter": {"id": "ent-e3-ctn-a", "type": "entity", "title": "A", "entity_kind": "test"}},
                req_id=321,
            ),
            _call_tool(
                async_client, sid, "write_page",
                {"frontmatter": {"id": "ent-e3-ctn-b", "type": "entity", "title": "B", "entity_kind": "test"}},
                req_id=322,
            ),
        )

        after = await _call_tool(async_client, sid, "index_status", {}, req_id=323)

    delta = after["mutex"]["acquire_count"] - baseline_count
    assert delta >= 2, f"acquire_count advanced by {delta}, expected ≥ 2"
    # mean_wait_ms across all acquires (including any past ones from
    # earlier tests in same module) should be > 0 because at least
    # one of our two writes had to wait.
    assert after["mutex"]["mean_wait_ms"] > 0, (
        "mean_wait_ms is 0 — contention counters not tracking, or no actual contention"
    )
