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
    # Server defaults to read_write scope; every tool (7 read-only + 4
    # read-write) should be listed.
    assert names == {
        # READ_ONLY
        "status",
        "list_pages",
        "read_page",
        "traverse",
        "search",
        "list_domains",
        "list_proposals",
        # READ_WRITE
        "bootstrap",
        "write_page",
        "write_pages",
        "write_proposal",
        "add_link",
        "add_claim",
    }


def test_status_tool(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "status", {}, req_id=10)
    # smalt_dir was set by conftest.py before the server imported config; the
    # seed-Smalt fixture also created the LanceDB tables and indexed 6 pages.
    assert result["exists"] is True
    assert set(result["tables"]) >= {"pages", "embeddings", "links", "claims", "sources"}
    # Lower bound — write_page / write_pages tests add more during the session.
    assert result["page_count"] >= 6
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
    # Seed includes 6 pages; write_page tests can add more — assert seed subset.
    ids = {p["id"] for p in result["pages"]}
    assert {"ent-alice", "ent-bob", "con-cs", "con-embedding", "con-index", "src-doc1"} <= ids


def test_list_pages_filter_by_type(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_pages", {"type": "entity"}, req_id=21)
    ids = {p["id"] for p in result["pages"]}
    # ent-alice + ent-bob from seed; write_page tests may add more entities.
    assert {"ent-alice", "ent-bob"} <= ids
    assert all(p["type"] == "entity" for p in result["pages"])


def test_list_pages_filter_by_prefix(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_pages", {"prefix": "con-"}, req_id=22)
    ids = {p["id"] for p in result["pages"]}
    assert ids == {"con-cs", "con-embedding", "con-index"}


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


# ---------------------------------------------------------------------------
# list_domains


def test_list_domains_finds_seed_domain(mcp_client: TestClient):
    """The seed Smalt's con-cs concept is flagged is_domain: true."""
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_domains", {}, req_id=90)
    ids = {d["id"] for d in result["domains"]}
    assert "con-cs" in ids
    for d in result["domains"]:
        assert {"id", "title", "path"} <= d.keys()


def test_list_domains_ignores_non_domain_concepts(mcp_client: TestClient):
    """con-embedding has glossary: true but is_domain: false — must NOT appear."""
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "list_domains", {}, req_id=91)
    ids = {d["id"] for d in result["domains"]}
    assert "con-embedding" not in ids
    assert "con-index" not in ids


# ---------------------------------------------------------------------------
# write_proposal + list_proposals


def _proposal_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def test_write_proposal_routes_schema_kind_to_schema_subdir(mcp_client: TestClient):
    """schema_addition proposals land in tasks/proposals/schema/, regardless of proposed_by."""
    sid = _initialize(mcp_client)
    fm = {
        "id": "prop-test-schema-1",
        "type": "proposal",
        "title": "Add foo field to ConceptPage",
        "proposal_kind": "schema_addition",
        "proposed_by": "cogitate",
        "proposed_at": _proposal_now(),
    }
    result = _call_tool(
        mcp_client,
        sid,
        "write_proposal",
        {"frontmatter": fm, "body": "## Observation\n\nfoo"},
        req_id=100,
    )
    assert result["id"] == "prop-test-schema-1"
    assert result["subdir"] == "schema"
    assert result["path"] == "tasks/proposals/schema/prop-test-schema-1.md"
    assert result["proposal_kind"] == "schema_addition"
    assert result["status"] == "proposed"


def test_write_proposal_routes_other_kind_to_proposer_subdir(mcp_client: TestClient):
    """Non-schema kinds land in tasks/proposals/<proposed_by>/."""
    sid = _initialize(mcp_client)
    fm = {
        "id": "prop-test-research-1",
        "type": "proposal",
        "title": "Ingest the foo paper",
        "proposal_kind": "source_adoption",
        "proposed_by": "research",
        "proposed_at": _proposal_now(),
    }
    result = _call_tool(
        mcp_client,
        sid,
        "write_proposal",
        {"frontmatter": fm},
        req_id=101,
    )
    assert result["subdir"] == "research"
    assert result["path"] == "tasks/proposals/research/prop-test-research-1.md"


def test_write_proposal_validation_error(mcp_client: TestClient):
    """Missing required field → structured validation_error."""
    sid = _initialize(mcp_client)
    fm = {
        "id": "prop-test-bad",
        "type": "proposal",
        "title": "Bad",
        # missing proposal_kind, proposed_by, proposed_at
    }
    result = _call_tool(
        mcp_client, sid, "write_proposal", {"frontmatter": fm}, req_id=102
    )
    assert result["error"] == "validation_error"


def test_list_proposals_includes_written(mcp_client: TestClient):
    """After write_proposal, list_proposals should surface the new entry."""
    sid = _initialize(mcp_client)
    # Write a fresh proposal so we can find it by id below
    proposal_id = "prop-test-list-1"
    fm = {
        "id": proposal_id,
        "type": "proposal",
        "title": "Listable proposal",
        "proposal_kind": "wiki_edge",
        "proposed_by": "cogitate",
        "proposed_at": _proposal_now(),
    }
    _call_tool(mcp_client, sid, "write_proposal", {"frontmatter": fm}, req_id=110)

    result = _call_tool(mcp_client, sid, "list_proposals", {}, req_id=111)
    by_id = {p["id"]: p for p in result["proposals"]}
    assert proposal_id in by_id
    entry = by_id[proposal_id]
    assert entry["proposal_kind"] == "wiki_edge"
    assert entry["proposed_by"] == "cogitate"
    assert entry["subdir"] == "cogitate"
    assert entry["status"] == "proposed"
    assert entry["path"].endswith(f"cogitate/{proposal_id}.md")


def test_list_proposals_filter_by_system(mcp_client: TestClient):
    """system filter narrows to one subdir."""
    sid = _initialize(mcp_client)
    # Ensure at least one schema and one research proposal exist
    for fm, body in [
        (
            {
                "id": "prop-test-filter-schema",
                "type": "proposal",
                "title": "Schema add",
                "proposal_kind": "schema_addition",
                "proposed_by": "cogitate",
                "proposed_at": _proposal_now(),
            },
            "",
        ),
        (
            {
                "id": "prop-test-filter-research",
                "type": "proposal",
                "title": "Source adopt",
                "proposal_kind": "source_adoption",
                "proposed_by": "research",
                "proposed_at": _proposal_now(),
            },
            "",
        ),
    ]:
        _call_tool(
            mcp_client, sid, "write_proposal", {"frontmatter": fm, "body": body}, req_id=120
        )

    res_schema = _call_tool(
        mcp_client, sid, "list_proposals", {"system": "schema"}, req_id=121
    )
    schema_ids = {p["id"] for p in res_schema["proposals"]}
    assert "prop-test-filter-schema" in schema_ids
    assert "prop-test-filter-research" not in schema_ids
    assert all(p["subdir"] == "schema" for p in res_schema["proposals"])

    res_research = _call_tool(
        mcp_client, sid, "list_proposals", {"system": "research"}, req_id=122
    )
    research_ids = {p["id"] for p in res_research["proposals"]}
    assert "prop-test-filter-research" in research_ids
    assert "prop-test-filter-schema" not in research_ids


def test_list_proposals_filter_by_kind(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "list_proposals", {"kind": "source_adoption"}, req_id=130
    )
    assert all(p["proposal_kind"] == "source_adoption" for p in result["proposals"])
    # At least the research filter test wrote one of these.
    assert result["count"] >= 1


def test_list_proposals_filter_by_status(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    # Every test proposal has default status="proposed"
    result = _call_tool(
        mcp_client, sid, "list_proposals", {"status": "proposed"}, req_id=131
    )
    assert all(p["status"] == "proposed" for p in result["proposals"])
    assert result["count"] >= 1
    # No "applied" proposals exist in this test session
    none_applied = _call_tool(
        mcp_client, sid, "list_proposals", {"status": "applied"}, req_id=132
    )
    assert none_applied["count"] == 0


# ---------------------------------------------------------------------------
# add_link


def test_add_link_to_existing_page_round_trips_via_traverse(mcp_client: TestClient):
    """Adding a link from ent-alice → ent-bob should surface in traverse()."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": "ent-alice", "to_id": "ent-bob", "label": "friend_of"},
        req_id=200,
    )
    # First time the friend_of edge is added in this session; should succeed.
    # (If another test added it first, response is added=False/reason=duplicate;
    # the post-condition — traverse sees the edge — still holds.)
    assert result.get("added") is True or result.get("reason") == "duplicate"
    # Round-trip via traverse with label filter.
    trav = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "ent-alice", "label": "friend_of"},
        req_id=201,
    )
    targets = {e["to_id"] for e in trav["edges"]}
    assert "ent-bob" in targets


def test_add_link_duplicate_is_no_op(mcp_client: TestClient):
    """Same (target, label) twice → second call returns added=False, reason=duplicate."""
    sid = _initialize(mcp_client)
    # First add (may or may not be the first time, depending on test order).
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": "ent-alice", "to_id": "ent-bob", "label": "knows"},
        req_id=210,
    )
    # Second add of the same edge — must be detected as duplicate.
    result = _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": "ent-alice", "to_id": "ent-bob", "label": "knows"},
        req_id=211,
    )
    assert result["added"] is False
    assert result["reason"] == "duplicate"


def test_add_link_unknown_page(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": "ent-does-not-exist", "to_id": "ent-bob", "label": "x"},
        req_id=212,
    )
    assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# add_claim


def test_add_claim_to_existing_page_round_trips_via_read_page(mcp_client: TestClient):
    """Adding a claim shows up in read_page's frontmatter.claims."""
    sid = _initialize(mcp_client)
    claim = {
        "id": "claim-test-1",
        "text": "Alice has a fictional age of 30",
        "value_type": "number",
        "value": 30,
        "unit": "years",
        "confidence": 0.5,
        "confidence_label": "medium",
    }
    result = _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": "ent-alice", "claim": claim},
        req_id=220,
    )
    assert result.get("added") is True or result.get("reason") == "duplicate_claim_id"
    # Round-trip via read_page — claim should be in frontmatter.
    page = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-alice"}, req_id=221
    )
    claim_ids = {c.get("id") for c in (page["frontmatter"].get("claims") or [])}
    assert "claim-test-1" in claim_ids


def test_add_claim_duplicate_id_is_no_op(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    claim = {"id": "claim-dup-test", "text": "first"}
    _call_tool(
        mcp_client, sid, "add_claim", {"page_id": "ent-alice", "claim": claim}, req_id=230
    )
    # Same id, different text — still rejected as duplicate.
    result = _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": "ent-alice", "claim": {"id": "claim-dup-test", "text": "second"}},
        req_id=231,
    )
    assert result["added"] is False
    assert result["reason"] == "duplicate_claim_id"


def test_add_claim_mcp_rejects_missing_required(mcp_client: TestClient):
    """MCP-side: the inner claim object's `required: [id, text]` is enforced
    before dispatch. Missing `text` gets a plain-text MCP error."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": "ent-alice", "claim": {"id": "claim-bad-required"}},
        req_id=240,
    )
    assert "text" in text and ("required" in text.lower() or "missing" in text.lower())


def test_add_claim_handler_rejects_invalid_value(mcp_client: TestClient):
    """Handler-side: Pydantic's Claim model enforces 0 <= confidence <= 1.
    MCP's input schema doesn't check that — so this exercises the handler's
    `Claim.model_validate` path and returns a structured validation_error."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {
            "page_id": "ent-alice",
            "claim": {"id": "claim-bad-conf", "text": "out of range", "confidence": 1.5},
        },
        req_id=245,
    )
    assert result["error"] == "validation_error"


def test_add_claim_unknown_page(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": "ent-nope", "claim": {"id": "claim-x", "text": "x"}},
        req_id=241,
    )
    assert result["error"] == "not_found"
