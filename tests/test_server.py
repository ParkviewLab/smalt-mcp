"""End-to-end smoke test: server starts; HTTP endpoints respond; MCP `status` round-trips."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _parse_sse(body: str) -> list[dict]:
    """Pull `data: <json>` payloads out of a Streamable HTTP SSE response."""
    out: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
        elif line.startswith("data:"):
            out.append(json.loads(line[5:]))
    return out


def _mcp(
    client: TestClient,
    method: str,
    params: dict | None = None,
    *,
    req_id: int = 1,
    session_id: str | None = None,
) -> tuple[dict, dict]:
    """Send a JSON-RPC request to /sse; return (response_json, response_headers)."""
    payload: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    resp = client.post("/sse", json=payload, headers=headers)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text!r}"
    ct = resp.headers.get("content-type", "")
    if ct.startswith("application/json"):
        return resp.json(), dict(resp.headers)
    msgs = _parse_sse(resp.text)
    assert msgs, f"no SSE data lines in {resp.text!r}"
    return msgs[-1], dict(resp.headers)


def _initialize(client: TestClient) -> str:
    body, headers = _mcp(
        client,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0.0"},
        },
        req_id=1,
    )
    assert body.get("result", {}).get("serverInfo", {}).get("name") == "smalt-mcp"
    return headers.get("mcp-session-id", "")


def _call_tool(
    client: TestClient,
    session_id: str,
    name: str,
    arguments: dict,
    *,
    req_id: int = 100,
) -> dict:
    body, _ = _mcp(
        client,
        "tools/call",
        {"name": name, "arguments": arguments},
        req_id=req_id,
        session_id=session_id,
    )
    assert "result" in body, f"tools/call returned: {body!r}"
    contents = body["result"]["content"]
    assert contents and contents[0]["type"] == "text"
    return json.loads(contents[0]["text"])


def _call_tool_raw_text(
    client: TestClient,
    session_id: str,
    name: str,
    arguments: dict,
    *,
    req_id: int = 100,
) -> str:
    """Return the first content block's raw text — without JSON-parsing.

    Used for tests that hit MCP's input-schema validation (which rejects
    with a plain-text error message, not a JSON payload).
    """
    body, _ = _mcp(
        client,
        "tools/call",
        {"name": name, "arguments": arguments},
        req_id=req_id,
        session_id=session_id,
    )
    assert "result" in body, f"tools/call returned: {body!r}"
    contents = body["result"]["content"]
    assert contents and contents[0]["type"] == "text"
    return contents[0]["text"]


# ---------------------------------------------------------------------------
# HTTP routes


def test_health(mcp_client: TestClient):
    r = mcp_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["version"]
    assert body["uptime_seconds"] >= 0


def test_admin_version(mcp_client: TestClient):
    r = mcp_client.get("/admin/version")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "smalt-mcp"
    assert body["version"]
    assert body["scope"] in {"read_only", "read_write"}
    assert body["smalt_dir"]


# ---------------------------------------------------------------------------
# MCP surface


def test_mcp_initialize_lists_all_tools(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    body, _ = _mcp(mcp_client, "tools/list", {}, req_id=2, session_id=sid)
    assert "result" in body, f"tools/list returned: {body!r}"
    names = {t["name"] for t in body["result"]["tools"]}
    # Server defaults to read_write scope; every tool (5 read-only + 3
    # read-write) should be listed. If scope is tightened later via
    # SMALT_SCOPE=read_only, the read-write set drops out.
    assert names == {
        "status",
        "list_pages",
        "read_page",
        "traverse",
        "search",
        "bootstrap",
        "write_page",
        "write_pages",
    }


def test_status_tool(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "status", {}, req_id=10)
    # smalt_dir was set by conftest.py before the server imported config; the
    # seed-Smalt fixture also created the LanceDB tables and indexed 5 pages.
    assert result["exists"] is True
    assert set(result["tables"]) >= {"pages", "embeddings", "links", "claims", "sources"}
    assert result["page_count"] == 5
    assert result["embedding"]["provider"] == "fake"
    assert result["embedding"]["dim"] == 384


def test_unknown_tool_returns_structured_error(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "does_not_exist", {}, req_id=11)
    assert result.get("error") == "unknown_tool"


# ---------------------------------------------------------------------------
# list_pages


def test_list_pages_all(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_pages", {}, req_id=20)
    assert result["count"] == 5
    ids = {p["id"] for p in result["pages"]}
    assert ids == {"ent-alice", "ent-bob", "con-embedding", "con-index", "src-doc1"}


def test_list_pages_filter_by_type(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_pages", {"type": "entity"}, req_id=21)
    ids = {p["id"] for p in result["pages"]}
    assert ids == {"ent-alice", "ent-bob"}
    assert all(p["type"] == "entity" for p in result["pages"])


def test_list_pages_filter_by_prefix(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_pages", {"prefix": "con-"}, req_id=22)
    ids = {p["id"] for p in result["pages"]}
    assert ids == {"con-embedding", "con-index"}


# ---------------------------------------------------------------------------
# read_page


def test_read_page_existing(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "read_page", {"page_id": "ent-alice"}, req_id=30)
    assert result["id"] == "ent-alice"
    assert result["title"] == "Alice"
    assert result["type"] == "entity"
    assert "Alice is a fictional person" in result["body"]
    # Frontmatter is round-tripped JSON; basic shape check.
    assert result["frontmatter"]["type"] == "entity"
    assert "Alicia" in result["frontmatter"].get("aliases", [])


def test_read_page_missing(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "read_page", {"page_id": "ent-nope"}, req_id=31)
    assert result.get("error") == "not_found"
    assert result.get("page_id") == "ent-nope"


def test_read_page_requires_page_id(mcp_client: TestClient):
    """MCP's input-schema validation rejects calls missing a `required` field."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(mcp_client, sid, "read_page", {}, req_id=32)
    assert "page_id" in text and ("required" in text.lower() or "missing" in text.lower())


# ---------------------------------------------------------------------------
# traverse


def test_traverse_one_hop(mcp_client: TestClient):
    """con-index --built_over--> con-embedding (1-hop)."""
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "traverse", {"from_id": "con-index"}, req_id=40)
    assert result["from_id"] == "con-index"
    assert result["count"] == 1
    edge = result["edges"][0]
    assert edge["from_id"] == "con-index"
    assert edge["to_id"] == "con-embedding"
    assert edge["label"] == "built_over"


def test_traverse_with_label_filter(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    # con-embedding --example_of--> ent-alice
    result = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "con-embedding", "label": "example_of"},
        req_id=41,
    )
    assert result["count"] == 1
    assert result["edges"][0]["to_id"] == "ent-alice"

    # Label that no edge has → empty.
    result = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "con-embedding", "label": "nonexistent"},
        req_id=42,
    )
    assert result["count"] == 0


def test_traverse_no_outgoing(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "traverse", {"from_id": "ent-alice"}, req_id=43)
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# search


def test_search_finds_relevant_pages(mcp_client: TestClient):
    """The word 'embedding' appears in 3 of 5 seed pages — FTS finds them."""
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "search", {"query": "embedding"}, req_id=50)
    ids = {r["id"] for r in result["results"]}
    # At minimum these three should rank in the top results (FTS body match).
    # The fake embedder may pull in others via random vector similarity, so
    # we assert subset rather than equality.
    assert {"con-embedding", "con-index", "src-doc1"} <= ids
    # Snippet + score shape:
    for r in result["results"]:
        assert "snippet" in r
        assert "score" in r
        assert r["score"] > 0


def test_search_top_k_caps_results(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "search", {"query": "Alice", "top_k": 2}, req_id=51)
    assert result["count"] <= 2


def test_search_requires_query(mcp_client: TestClient):
    """MCP's input-schema validation rejects calls missing a `required` field."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(mcp_client, sid, "search", {}, req_id=52)
    assert "query" in text and ("required" in text.lower() or "missing" in text.lower())


# ---------------------------------------------------------------------------
# bootstrap


def test_bootstrap_idempotent(mcp_client: TestClient):
    """Second bootstrap call must be a complete no-op."""
    sid = _initialize(mcp_client)
    # First call: may create some of the dirs/files the conftest didn't —
    # we don't assert specifics, just that the call succeeds and returns
    # the expected shape.
    first = _call_tool(mcp_client, sid, "bootstrap", {}, req_id=60)
    assert first["smalt_dir"]
    assert isinstance(first["created_dirs"], list)
    assert isinstance(first["created_files"], list)
    assert isinstance(first["created_tables"], list)

    # Second call: every canonical dir / file / table is already in place.
    second = _call_tool(mcp_client, sid, "bootstrap", {}, req_id=61)
    assert second["created_dirs"] == []
    assert second["created_files"] == []
    assert second["created_tables"] == []


# ---------------------------------------------------------------------------
# write_page


def test_write_page_round_trip(mcp_client: TestClient):
    """write_page → read_page returns the same content."""
    sid = _initialize(mcp_client)
    fm = {
        "id": "ent-test-write-rt",
        "type": "entity",
        "title": "Test Write Roundtrip",
        "entity_kind": "test",
    }
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": fm, "body": "roundtrip body content"},
        req_id=70,
    )
    assert result["id"] == "ent-test-write-rt"
    assert result["path"] == "pages/entities/ent-test-write-rt.md"
    assert result["type"] == "entity"
    assert "index_result" in result
    # Indexer should have picked up exactly one new page.
    idx = result["index_result"]
    assert idx["inserted"] >= 1
    # Round-trip via read_page.
    read = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-test-write-rt"}, req_id=71
    )
    assert read["title"] == "Test Write Roundtrip"
    assert read["body"] == "roundtrip body content"
    assert read["type"] == "entity"


def test_write_page_validation_error(mcp_client: TestClient):
    """Frontmatter missing a required field → structured validation_error."""
    sid = _initialize(mcp_client)
    # Missing `title` — entity requires id, type, title at minimum.
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": {"id": "ent-test-invalid", "type": "entity"}},
        req_id=72,
    )
    assert result["error"] == "validation_error"


def test_write_page_overwrite_updates(mcp_client: TestClient):
    """Writing the same id twice updates the page; the second body wins."""
    sid = _initialize(mcp_client)
    fm = {"id": "ent-test-write-ow", "type": "entity", "title": "v1", "entity_kind": "test"}
    _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm, "body": "first"}, req_id=73)
    fm2 = {**fm, "title": "v2"}
    _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm2, "body": "second"}, req_id=74)
    read = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-test-write-ow"}, req_id=75
    )
    assert read["title"] == "v2"
    assert read["body"] == "second"


# ---------------------------------------------------------------------------
# write_pages


def test_write_pages_batch(mcp_client: TestClient):
    """Two pages written in one batch are both indexed and readable."""
    sid = _initialize(mcp_client)
    pages = [
        {
            "frontmatter": {
                "id": "ent-test-batch-1",
                "type": "entity",
                "title": "Batch 1",
                "entity_kind": "test",
            },
            "body": "batch page 1",
        },
        {
            "frontmatter": {
                "id": "ent-test-batch-2",
                "type": "entity",
                "title": "Batch 2",
                "entity_kind": "test",
            },
            "body": "batch page 2",
        },
    ]
    result = _call_tool(mcp_client, sid, "write_pages", {"pages": pages}, req_id=80)
    assert result["count"] == 2
    written_ids = {w["id"] for w in result["written"]}
    assert written_ids == {"ent-test-batch-1", "ent-test-batch-2"}
    # Both readable.
    for pid in written_ids:
        r = _call_tool(mcp_client, sid, "read_page", {"page_id": pid}, req_id=81)
        assert r["id"] == pid


def test_write_pages_validation_aborts_batch(mcp_client: TestClient):
    """Invalid entry N aborts the whole batch — earlier valid entries are NOT written."""
    sid = _initialize(mcp_client)
    pages = [
        {
            "frontmatter": {
                "id": "ent-test-abort-ok",
                "type": "entity",
                "title": "OK",
                "entity_kind": "test",
            },
            "body": "",
        },
        # Second: missing required `title`
        {"frontmatter": {"id": "ent-test-abort-bad", "type": "entity"}, "body": ""},
    ]
    result = _call_tool(mcp_client, sid, "write_pages", {"pages": pages}, req_id=82)
    assert result["error"] == "validation_error"
    assert result["index"] == 1
    # The valid first page was NOT written (validate-all-then-write contract).
    r = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-test-abort-ok"}, req_id=83
    )
    assert r.get("error") == "not_found"
