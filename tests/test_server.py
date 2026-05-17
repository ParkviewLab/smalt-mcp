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
    assert body["scope"] in {"read_only", "read_write", "remove_destructive"}
    assert body["smalt_dir"]


# ---------------------------------------------------------------------------
# MCP surface


def test_mcp_initialize_lists_all_tools(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    body, _ = _mcp(mcp_client, "tools/list", {}, req_id=2, session_id=sid)
    assert "result" in body, f"tools/list returned: {body!r}"
    names = {t["name"] for t in body["result"]["tools"]}
    # Server runs at remove_destructive scope (from conftest); every tool
    # (8 read-only + 5 read-write + 4 remove-destructive = 17) should be
    # listed. Proposal / experiment / gap tools moved to ebony-enriching.
    assert names == {
        # READ_ONLY (8)
        "status",
        "list_pages",
        "read_page",
        "find_by_alias",
        "incoming_links",
        "traverse",
        "search",
        "list_domains",
        # READ_WRITE (5)
        "bootstrap",
        "write_page",
        "write_pages",
        "add_link",
        "add_claim",
        # REMOVE_DESTRUCTIVE (4)
        "remove_page",
        "update_claim",
        "remove_claim",
        "remove_link",
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


# ---- property filters on list_pages (C-2) ----


def test_list_pages_filter_by_glossary(mcp_client: TestClient):
    """Seed: con-embedding + con-index both have `glossary: true`; con-cs has
    `is_domain: true` (and no glossary field)."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "list_pages", {"glossary": True}, req_id=110
    )
    ids = {p["id"] for p in result["pages"]}
    assert {"con-embedding", "con-index"} <= ids
    assert "con-cs" not in ids  # is_domain, not glossary
    assert "ent-alice" not in ids  # not a concept
    assert result["truncated"] is False


def test_list_pages_filter_by_is_domain(mcp_client: TestClient):
    """Seed: con-cs is the only `is_domain: true` page."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "list_pages", {"is_domain": True}, req_id=111
    )
    ids = {p["id"] for p in result["pages"]}
    assert "con-cs" in ids
    assert "con-embedding" not in ids
    assert "con-index" not in ids


def test_list_pages_filter_by_domain(mcp_client: TestClient):
    """Seed: ent-alice, con-embedding, con-index, src-doc1 all tag `domains: [con-cs]`."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "list_pages", {"domain": "con-cs"}, req_id=112
    )
    ids = {p["id"] for p in result["pages"]}
    assert {"ent-alice", "con-embedding", "con-index", "src-doc1"} <= ids
    # ent-bob has no `domains:` field; con-cs doesn't self-tag.
    assert "ent-bob" not in ids
    assert "con-cs" not in ids


def test_list_pages_filter_by_has_aliases_containing(mcp_client: TestClient):
    """Seed: ent-alice has `aliases: [Alicia]`."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "list_pages",
        {"has_aliases_containing": "Alicia"},
        req_id=113,
    )
    ids = {p["id"] for p in result["pages"]}
    assert "ent-alice" in ids
    assert "ent-bob" not in ids


def test_list_pages_filter_by_fetched_at_range(mcp_client: TestClient):
    """Write two SourcePages with explicit fetched_at; filter by both endpoints
    of the date range. (Seed src-doc1 has no fetched_at — won't match either
    fetched_at filter under SQL-NULL semantics.)"""
    sid = _initialize(mcp_client)
    # Write an old source.
    _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {
                "id": "src-fetched-2020",
                "type": "source",
                "title": "Old source",
                "location_uri": "file:/tmp/old.md",
                "location_kind": "file",
                "fetched_at": "2020-01-15T00:00:00",
            },
        },
        req_id=114,
    )
    # Write a new source.
    _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {
                "id": "src-fetched-2025",
                "type": "source",
                "title": "New source",
                "location_uri": "file:/tmp/new.md",
                "location_kind": "file",
                "fetched_at": "2025-08-01T00:00:00",
            },
        },
        req_id=115,
    )

    # fetched_at_before=2024 → old source only.
    result = _call_tool(
        mcp_client,
        sid,
        "list_pages",
        {"fetched_at_before": "2024-01-01T00:00:00"},
        req_id=116,
    )
    original_ids = set()
    for p in result["pages"]:
        # write_page mangles ids; the original is in aliases. We don't need
        # the original here — we just check that the old source's canonical
        # id (which starts with the caller id) is in the result.
        original_ids.add(p["id"])
    matched_old = [pid for pid in original_ids if pid.startswith("src-fetched-2020-")]
    matched_new = [pid for pid in original_ids if pid.startswith("src-fetched-2025-")]
    assert len(matched_old) >= 1
    assert len(matched_new) == 0
    # Seed src-doc1 has no fetched_at, so SQL-NULL semantics excludes it.
    assert "src-doc1" not in original_ids

    # fetched_at_after=2024 → new source only.
    result = _call_tool(
        mcp_client,
        sid,
        "list_pages",
        {"fetched_at_after": "2024-01-01T00:00:00"},
        req_id=117,
    )
    ids = {p["id"] for p in result["pages"]}
    matched_old = [pid for pid in ids if pid.startswith("src-fetched-2020-")]
    matched_new = [pid for pid in ids if pid.startswith("src-fetched-2025-")]
    assert len(matched_old) == 0
    assert len(matched_new) >= 1


def test_list_pages_filter_combined_type_and_glossary(mcp_client: TestClient):
    """AND-composed: type=concept AND glossary=True returns the two glossary
    concepts only (excludes con-cs which is a concept but not glossary)."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "list_pages",
        {"type": "concept", "glossary": True},
        req_id=118,
    )
    ids = {p["id"] for p in result["pages"]}
    assert {"con-embedding", "con-index"} <= ids
    assert "con-cs" not in ids
    assert all(p["type"] == "concept" for p in result["pages"])


def test_list_pages_filter_validation_error_on_bad_datetime(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "list_pages",
        {"fetched_at_before": "not-a-real-date"},
        req_id=119,
    )
    assert result["error"] == "validation_error"
    assert "fetched_at_before" in result["message"]


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
    assert result["hops"] == 1
    edge = result["edges"][0]
    assert edge["from_id"] == "con-index"
    assert edge["to_id"] == "con-embedding"
    assert edge["label"] == "built_over"
    # Multi-hop fields present even at 1-hop.
    assert set(result["visited_nodes"]) == {"con-index", "con-embedding"}
    assert result["truncated"] is False


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
    # Even with no outgoing edges, visited_nodes contains the seed.
    assert result["visited_nodes"] == ["ent-alice"]


# ---- multi-hop ----


def test_traverse_two_hops_walks_chain(mcp_client: TestClient):
    """con-index --built_over--> con-embedding --example_of--> ent-alice.

    A 2-hop walk from con-index should collect BOTH edges and visit all
    three nodes. The seed Smalt's link chain (set up in conftest.py) is
    deterministic, so we can assert exact counts.
    """
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "traverse", {"from_id": "con-index", "hops": 2}, req_id=44
    )
    assert result["hops"] == 2
    # Two edges total: con-index->con-embedding (hop 1) + con-embedding->ent-alice (hop 2).
    assert result["count"] == 2
    edge_keys = {(e["from_id"], e["to_id"], e["label"]) for e in result["edges"]}
    assert ("con-index", "con-embedding", "built_over") in edge_keys
    assert ("con-embedding", "ent-alice", "example_of") in edge_keys
    # Visited contains seed + both reached nodes.
    assert set(result["visited_nodes"]) == {"con-index", "con-embedding", "ent-alice"}


def test_traverse_label_filter_applied_per_hop(mcp_client: TestClient):
    """label=built_over walks con-index->con-embedding but stops there
    (con-embedding's outgoing edge to ent-alice is labeled example_of,
    not built_over). With hops=3 the walk still terminates at hop 1."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "con-index", "hops": 3, "label": "built_over"},
        req_id=45,
    )
    assert result["count"] == 1
    assert result["edges"][0]["to_id"] == "con-embedding"
    # Walk stopped: ent-alice is reachable in 2 hops but only via
    # example_of, which the per-hop filter excludes.
    assert set(result["visited_nodes"]) == {"con-index", "con-embedding"}
    assert "ent-alice" not in result["visited_nodes"]


def test_traverse_handles_cycle_without_infinite_loop(mcp_client: TestClient):
    """Build a cycle A->B->A and walk with hops=5; BFS visited-set prevents
    re-expansion. Both edges should appear; no duplicate node visits."""
    sid = _initialize(mcp_client)
    # Create the cycle: two new entity pages + reciprocal links.
    a = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-cycle-a", "cycle a")},
        req_id=46,
    )
    b = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-cycle-b", "cycle b")},
        req_id=47,
    )
    a_id, b_id = a["id"], b["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": a_id, "to_id": b_id, "label": "next"},
        req_id=48,
    )
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": b_id, "to_id": a_id, "label": "next"},
        req_id=49,
    )
    # Walk from A with hops well beyond the cycle length.
    result = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": a_id, "hops": 5, "label": "next"},
        req_id=50,
    )
    # Two distinct edges total (A->B and B->A); both should be present.
    edge_keys = {(e["from_id"], e["to_id"]) for e in result["edges"]}
    assert (a_id, b_id) in edge_keys
    assert (b_id, a_id) in edge_keys
    assert result["count"] == 2  # exactly two edges, not more (no infinite loop)
    assert set(result["visited_nodes"]) == {a_id, b_id}


def test_traverse_rejects_hops_too_high(mcp_client: TestClient):
    """hops > 5 is rejected at the MCP input-schema layer (the tool spec
    declares `maximum: 5`). MCP returns a plain-text error before the
    handler runs."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "con-index", "hops": 99},
        req_id=51,
    )
    assert "maximum" in text.lower() or "5" in text


def test_traverse_rejects_hops_too_low(mcp_client: TestClient):
    """hops < 1 is rejected at the MCP input-schema layer (the tool spec
    declares `minimum: 1`). The handler's defense-in-depth runtime check
    catches anything that slips past (e.g. if future MCP servers loosen
    the schema enforcement)."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(
        mcp_client,
        sid,
        "traverse",
        {"from_id": "con-index", "hops": 0},
        req_id=52,
    )
    assert "minimum" in text.lower() or "1" in text


# ---------------------------------------------------------------------------
# search


def test_search_finds_relevant_pages(mcp_client: TestClient):
    """The word 'embedding' appears in 3 of 5 seed pages — FTS finds them.

    Every hit must carry id, aliases, title, type, snippet, score.
    """
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "search", {"query": "embedding"}, req_id=50)
    ids = {r["id"] for r in result["results"]}
    # At minimum these three should rank in the top results (FTS body match).
    # The fake embedder may pull in others via random vector similarity, so
    # we assert subset rather than equality.
    assert {"con-embedding", "con-index", "src-doc1"} <= ids
    # Per-hit shape: required fields present.
    for r in result["results"]:
        assert {"id", "aliases", "title", "type", "snippet", "score"} <= r.keys()
        assert isinstance(r["aliases"], list)  # may be empty, but present
        assert r["score"] > 0


def test_search_includes_aliases_for_mangled_pages(mcp_client: TestClient):
    """A page created via write_page (always-mangle) has its caller-id in
    aliases; that alias must appear in search hits so a caller can find
    the page by its memorable handle."""
    sid = _initialize(mcp_client)
    # Create a page with a distinctive body so FTS will rank it.
    fm = _ent_fm("ent-search-alias-probe", "Search-Alias-Probe Entity")
    create = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": fm,
            "body": "search-alias-probe distinctive body content for FTS",
        },
        req_id=53,
    )
    canonical = create["id"]
    # Search by a distinctive word from the body.
    result = _call_tool(
        mcp_client, sid, "search", {"query": "search-alias-probe"}, req_id=54
    )
    # The mangled page should be in the results, with the caller-id in aliases.
    matched = next((r for r in result["results"] if r["id"] == canonical), None)
    assert matched is not None, f"expected canonical {canonical} in search results"
    assert "ent-search-alias-probe" in matched["aliases"]


def test_search_top_k_caps_results(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(mcp_client, sid, "search", {"query": "Alice", "top_k": 2}, req_id=51)
    assert result["count"] <= 2


def test_search_matches_by_exact_alias(mcp_client: TestClient):
    """Searching for the EXACT alias of a mangled page finds that page,
    even if the alias doesn't appear in the page's body or title."""
    sid = _initialize(mcp_client)
    # Create a page whose body deliberately doesn't mention the caller-id.
    create = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _ent_fm("ent-zzqx-aliasprobe", "Unrelated Title"),
            "body": "totally unrelated body content with no matching tokens",
        },
        req_id=55,
    )
    canonical = create["id"]
    assert "ent-zzqx-aliasprobe" in create["original_id"]
    # Search by the caller-id (which is now an alias on the page).
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "ent-zzqx-aliasprobe"},
        req_id=56,
    )
    ids = {r["id"] for r in result["results"]}
    assert canonical in ids, (
        "search must find a page by alias even when the alias doesn't "
        "appear in body or title"
    )
    # And the hit itself carries the matched alias in its aliases list.
    matched = next(r for r in result["results"] if r["id"] == canonical)
    assert "ent-zzqx-aliasprobe" in matched["aliases"]


def test_search_tokenized_query_matches_alias(mcp_client: TestClient):
    """A query that EMBEDS an alias (with other words around it) still
    matches the page via alias retrieval — the whitespace-tokenized
    matching catches it."""
    sid = _initialize(mcp_client)
    create = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _ent_fm("ent-qqrx-tokenprobe", "Unrelated"),
            "body": "another unrelated body with no matching content",
        },
        req_id=57,
    )
    canonical = create["id"]
    # Query has other words; alias is one of the tokens.
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "tell me about ent-qqrx-tokenprobe please"},
        req_id=58,
    )
    ids = {r["id"] for r in result["results"]}
    assert canonical in ids


def test_search_alias_match_independent_of_fts_and_vector(mcp_client: TestClient):
    """A query that matches NEITHER body FTS nor title FTS, but matches an
    alias, still surfaces the page via the alias retrieval source.

    Vector similarity will return SOMETHING via random fake-embedder
    similarity, so we can't assert FTS+vector returned nothing — we just
    assert the alias-matched page IS in the results, and ranks well
    enough to fit in top_k."""
    sid = _initialize(mcp_client)
    create = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _ent_fm("ent-qqrx-onlyalias", "title only here"),
            "body": "body content has nothing in common with the query",
        },
        req_id=59,
    )
    canonical = create["id"]
    # Query is the bare alias — should rank highly due to alias match.
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "ent-qqrx-onlyalias", "top_k": 5},
        req_id=60,
    )
    ids = {r["id"] for r in result["results"]}
    assert canonical in ids


def test_search_requires_query(mcp_client: TestClient):
    """MCP's input-schema validation rejects calls missing a `required` field."""
    sid = _initialize(mcp_client)
    text = _call_tool_raw_text(mcp_client, sid, "search", {}, req_id=52)
    assert "query" in text and ("required" in text.lower() or "missing" in text.lower())


# ---- property filters on search (C-2) ----


def test_search_filter_by_glossary(mcp_client: TestClient):
    """search('embedding') over the seed surfaces con-embedding (glossary),
    con-index (glossary), and src-doc1 (source). With glossary=True, the
    source drops out and only the two concept-glossary entries remain."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "embedding", "glossary": True, "top_k": 10},
        req_id=180,
    )
    ids = {r["id"] for r in result["results"]}
    # The two glossary concepts must appear; the source must NOT.
    assert {"con-embedding", "con-index"} <= ids
    assert "src-doc1" not in ids
    # Every returned result must be a glossary concept (defense against
    # FTS/vector noise pulling in non-matching pages).
    for r in result["results"]:
        assert r["type"] == "concept"


def test_search_filter_by_domain(mcp_client: TestClient):
    """search('Alice') with domain=con-cs returns only pages tagged with
    that domain. ent-alice has domains:[con-cs] in the seed; ent-bob has
    no domains → excluded."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "Alice", "domain": "con-cs", "top_k": 10},
        req_id=181,
    )
    ids = {r["id"] for r in result["results"]}
    assert "ent-alice" in ids
    assert "ent-bob" not in ids


def test_search_filter_by_has_aliases_containing(mcp_client: TestClient):
    """ent-alice's `aliases: [Alicia]` makes it match
    has_aliases_containing='Alicia'."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "fictional person", "has_aliases_containing": "Alicia"},
        req_id=182,
    )
    ids = {r["id"] for r in result["results"]}
    assert "ent-alice" in ids
    assert "ent-bob" not in ids


def test_search_filter_no_matches_returns_empty(mcp_client: TestClient):
    """A filter that excludes every retrieved candidate returns an empty list
    (not an error)."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "embedding", "domain": "no-such-domain"},
        req_id=183,
    )
    assert result["count"] == 0
    assert result["results"] == []


def test_search_filter_validation_error_on_bad_datetime(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "search",
        {"query": "anything", "fetched_at_after": "not-a-real-date"},
        req_id=184,
    )
    assert result["error"] == "validation_error"
    assert "fetched_at_after" in result["message"]


# ---------------------------------------------------------------------------
# bootstrap


def test_bootstrap_idempotent(mcp_client: TestClient):
    """Second bootstrap call must be a complete no-op for dirs/files/tables.
    `index_result` is non-empty either way (the indexer runs to materialize
    or refresh the canonical IndexPages); just check the field exists."""
    sid = _initialize(mcp_client)
    # First call: may create some of the dirs/files the conftest didn't —
    # we don't assert specifics, just that the call succeeds and returns
    # the expected shape.
    first = _call_tool(mcp_client, sid, "bootstrap", {}, req_id=60)
    assert first["smalt_dir"]
    assert isinstance(first["created_dirs"], list)
    assert isinstance(first["created_files"], list)
    assert isinstance(first["created_tables"], list)
    # C-3: bootstrap now runs the indexer to materialize the IndexPages.
    assert isinstance(first["index_result"], dict)

    # Second call: every canonical dir / file / table is already in place;
    # IndexPages already exist with up-to-date content (since corpus
    # didn't change between the two bootstrap calls), so indexer skips
    # the regeneration write.
    second = _call_tool(mcp_client, sid, "bootstrap", {}, req_id=61)
    assert second["created_dirs"] == []
    assert second["created_files"] == []
    assert second["created_tables"] == []
    assert isinstance(second["index_result"], dict)


# ---------------------------------------------------------------------------
# IndexPage (C-3)


def test_bootstrap_materializes_canonical_index_pages(mcp_client: TestClient):
    """After bootstrap (which runs at session start via conftest), the two
    canonical IndexPages exist in the indexed page set with the expected ids
    + types + auto_generated flag."""
    sid = _initialize(mcp_client)
    # idx-glossary
    glossary = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "idx-glossary"}, req_id=420
    )
    assert glossary.get("error") is None
    assert glossary["type"] == "index"
    assert glossary["title"] == "Glossary"
    assert glossary["frontmatter"]["auto_generated"] is True
    assert glossary["frontmatter"]["stored_query"] == {
        "kind": "concept_flag",
        "flag": "glossary",
    }
    # The seed has con-embedding + con-index both flagged glossary: true
    # → body must reference both ids.
    assert "con-embedding" in glossary["body"]
    assert "con-index" in glossary["body"]

    # idx-domains
    domains = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "idx-domains"}, req_id=421
    )
    assert domains.get("error") is None
    assert domains["type"] == "index"
    assert domains["title"] == "Domains"
    assert domains["frontmatter"]["auto_generated"] is True
    assert domains["frontmatter"]["stored_query"] == {
        "kind": "concept_flag",
        "flag": "is_domain",
    }
    # Seed has only con-cs flagged is_domain: true.
    assert "con-cs" in domains["body"]


def test_index_pages_listed_with_type_filter(mcp_client: TestClient):
    """list_pages(type='index') returns exactly the two canonical IndexPages."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "list_pages", {"type": "index"}, req_id=422
    )
    ids = {p["id"] for p in result["pages"]}
    assert {"idx-glossary", "idx-domains"} <= ids
    assert all(p["type"] == "index" for p in result["pages"])


def test_index_page_regenerates_on_new_glossary_concept(mcp_client: TestClient):
    """Adding a `glossary: true` ConceptPage triggers the auto-indexer; the
    pages/glossary.md IndexPage's body should now reference the new concept."""
    sid = _initialize(mcp_client)
    # Add a new glossary concept.
    new_concept = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {
                "id": "con-regen-probe",
                "type": "concept",
                "title": "Regen Probe Concept",
                "glossary": True,
            },
            "body": "First sentence is the definition. Second sentence too.",
        },
        req_id=423,
    )
    canonical = new_concept["id"]
    # Read back the glossary IndexPage; it should now reference the new
    # canonical id in its body.
    glossary = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "idx-glossary"}, req_id=424
    )
    assert canonical in glossary["body"]
    # The first sentence should appear as the definition.
    assert "First sentence is the definition." in glossary["body"]


def test_index_page_skips_rewrite_when_corpus_unchanged(mcp_client: TestClient):
    """Bootstrap twice in succession — the second run must skip the
    IndexPage rewrite (content_hash stays stable across runs). We verify
    by reading the IndexPage and seeing its frontmatter is well-formed
    after a no-op-change indexer pass."""
    sid = _initialize(mcp_client)
    # First bootstrap (idempotent; runs the indexer).
    _call_tool(mcp_client, sid, "bootstrap", {}, req_id=425)
    before = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "idx-glossary"}, req_id=426
    )
    # Second bootstrap — no corpus change between them.
    _call_tool(mcp_client, sid, "bootstrap", {}, req_id=427)
    after = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "idx-glossary"}, req_id=428
    )
    # Body should be byte-identical (no rewrite happened).
    assert before["body"] == after["body"]
    # Frontmatter should also be identical.
    assert before["frontmatter"] == after["frontmatter"]


def test_write_page_rejects_index_page(mcp_client: TestClient):
    """Direct writes to type='index' are rejected (the indexer is the only
    writer for IndexPages)."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {
                "id": "idx-custom-attempt",
                "type": "index",
                "title": "Should Fail",
                "auto_generated": True,
                "stored_query": {"kind": "concept_flag", "flag": "glossary"},
            },
        },
        req_id=429,
    )
    assert result["error"] == "forbidden_page_type"
    assert result["type"] == "index"


def test_write_pages_batch_rejects_index_page_entry(mcp_client: TestClient):
    """One index entry in a batch aborts the whole batch (validate-all-then-act)."""
    sid = _initialize(mcp_client)
    pages_arg = [
        # An otherwise-valid concept page (would succeed on its own).
        {
            "frontmatter": _ent_fm("ent-batch-w-idx-ok", "OK"),
            "body": "",
        },
        # An IndexPage entry — should abort the batch at validation time.
        {
            "frontmatter": {
                "id": "idx-rejected-batch",
                "type": "index",
                "title": "Should Fail",
                "auto_generated": True,
                "stored_query": {"kind": "concept_flag", "flag": "glossary"},
            },
            "body": "",
        },
    ]
    result = _call_tool(
        mcp_client, sid, "write_pages", {"pages": pages_arg}, req_id=430
    )
    assert result["error"] == "forbidden_page_type"
    assert result["index"] == 1
    # The valid first entry must NOT have been written (all-or-nothing).
    probe = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-batch-w-idx-ok"}, req_id=431
    )
    assert probe.get("error") == "not_found"


# ---------------------------------------------------------------------------
# write_page


def test_write_page_round_trip(mcp_client: TestClient):
    """write_page (default mode=create) mangles the id, preserves the caller's
    id as an alias, and read_page on the canonical id round-trips the
    content. The path embeds the mangled canonical id."""
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
    # Canonical id is mangled: caller-id + '-' + 22-char base64.
    assert result["original_id"] == "ent-test-write-rt"
    assert result["id"].startswith("ent-test-write-rt-")
    assert len(result["id"]) == len("ent-test-write-rt") + 1 + 22
    assert result["mode"] == "create"
    assert result["mangled"] is True
    assert result["type"] == "entity"
    assert result["path"] == f"pages/entities/{result['id']}.md"
    assert result["index_result"]["inserted"] >= 1
    # Round-trip via read_page using the canonical id.
    read = _call_tool(
        mcp_client, sid, "read_page", {"page_id": result["id"]}, req_id=71
    )
    assert read["title"] == "Test Write Roundtrip"
    assert read["body"] == "roundtrip body content"
    assert read["type"] == "entity"
    # Caller's original id is preserved as an alias.
    assert "ent-test-write-rt" in read["frontmatter"].get("aliases", [])


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


def test_two_creates_with_same_caller_id_produce_distinct_canonical_ids(mcp_client: TestClient):
    """Always-mangle: two create calls with identical caller-id frontmatter
    produce two pages with different canonical ids; neither overwrites the
    other."""
    sid = _initialize(mcp_client)
    fm = {"id": "ent-test-twin", "type": "entity", "title": "v1", "entity_kind": "test"}
    r1 = _call_tool(
        mcp_client, sid, "write_page", {"frontmatter": fm, "body": "first"}, req_id=73
    )
    r2 = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": {**fm, "title": "v2"}, "body": "second"},
        req_id=74,
    )
    assert r1["original_id"] == r2["original_id"] == "ent-test-twin"
    assert r1["id"] != r2["id"], "two creates must produce distinct canonical ids"
    # Both files exist; both readable independently.
    read1 = _call_tool(mcp_client, sid, "read_page", {"page_id": r1["id"]}, req_id=75)
    read2 = _call_tool(mcp_client, sid, "read_page", {"page_id": r2["id"]}, req_id=76)
    assert read1["title"] == "v1"
    assert read1["body"] == "first"
    assert read2["title"] == "v2"
    assert read2["body"] == "second"
    # Both share the original id as an alias.
    assert "ent-test-twin" in read1["frontmatter"].get("aliases", [])
    assert "ent-test-twin" in read2["frontmatter"].get("aliases", [])


def test_write_page_update_overwrites_existing(mcp_client: TestClient):
    """Update mode against the canonical id from a prior create modifies that
    specific page in place."""
    sid = _initialize(mcp_client)
    fm = {"id": "ent-test-update", "type": "entity", "title": "v1", "entity_kind": "test"}
    create = _call_tool(
        mcp_client, sid, "write_page", {"frontmatter": fm, "body": "first"}, req_id=77
    )
    canonical = create["id"]
    # Update mode requires the canonical id in the frontmatter.
    update = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {**fm, "id": canonical, "title": "v2"},
            "body": "second",
            "mode": "update",
        },
        req_id=78,
    )
    assert update.get("error") is None, update
    assert update["mode"] == "update"
    assert update["mangled"] is False
    assert update["id"] == canonical
    # Body changed in place.
    read = _call_tool(mcp_client, sid, "read_page", {"page_id": canonical}, req_id=79)
    assert read["title"] == "v2"
    assert read["body"] == "second"


# ---------------------------------------------------------------------------
# write_pages


def test_write_pages_batch(mcp_client: TestClient):
    """Two pages written in one batch (mode=create by default) get mangled
    canonical ids; both are independently readable by their canonical ids."""
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
    originals = {w["original_id"] for w in result["written"]}
    assert originals == {"ent-test-batch-1", "ent-test-batch-2"}
    # All canonical ids are mangled and distinct.
    canonical_ids = {w["id"] for w in result["written"]}
    assert len(canonical_ids) == 2
    for w in result["written"]:
        assert w["mangled"] is True
        assert w["id"].startswith(w["original_id"] + "-")
    # Both readable by canonical id.
    for canonical in canonical_ids:
        r = _call_tool(mcp_client, sid, "read_page", {"page_id": canonical}, req_id=81)
        assert r["id"] == canonical


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


# ---------------------------------------------------------------------------
# ID validation (path traversal + portability)
#
# Schema-level: rejected ids should never reach the filesystem. Every test
# below also verifies that no file was created under pages/ (using
# list_pages or a directory probe via list_pages with a prefix filter).


def _ent_fm(page_id: str, title: str = "x") -> dict:
    return {"id": page_id, "type": "entity", "title": title, "entity_kind": "test"}


def test_write_page_rejects_path_traversal_id(mcp_client: TestClient):
    """`id: '../etc/passwd'` must be rejected at the schema layer."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("../etc/passwd")},
        req_id=300,
    )
    assert result["error"] == "validation_error"
    assert "id" in result["message"].lower()


def test_write_page_rejects_various_invalid_id_shapes(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    bad_ids = [
        "..",
        "has spaces",
        ".leading-dot",
        "-leading-hyphen",
        "_leading-underscore",
        "with/slash",
        "with\\backslash",
        "with:colon",
        "with<bracket",
        "",
    ]
    for i, bad in enumerate(bad_ids):
        result = _call_tool(
            mcp_client,
            sid,
            "write_page",
            {"frontmatter": _ent_fm(bad)},
            req_id=310 + i,
        )
        assert result["error"] == "validation_error", f"{bad!r} should have been rejected"


def test_write_page_rejects_windows_reserved_names(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    for i, bad in enumerate(["con", "NUL", "com1", "PRN", "lpt9"]):
        result = _call_tool(
            mcp_client,
            sid,
            "write_page",
            {"frontmatter": _ent_fm(bad)},
            req_id=330 + i,
        )
        assert result["error"] == "validation_error", f"{bad!r} should have been rejected"
        assert "windows" in result["message"].lower() or "reserved" in result["message"].lower()


# ---------------------------------------------------------------------------
# Section page id format (C-4)


def _src_fm_section(section_id: str, title: str = "section") -> dict:
    """Build a minimal SourcePage frontmatter for a section page."""
    return {
        "id": section_id,
        "type": "source",
        "title": title,
        "location_uri": f"file:/tmp/{section_id.replace('::', '__')}",
        "location_kind": "file",
    }


def test_section_id_round_trip(mcp_client: TestClient):
    """Write a section-shaped SourcePage; the canonical id is NOT mangled
    (it's already canonical by construction); the file lands at
    `pages/sources/<src>/<rel>.md`; read_page returns it."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section(
                "src-sec-rt::src/utils.py", "utils.py section"
            ),
            "body": "section body",
        },
        req_id=350,
    )
    assert result.get("error") is None, result
    # No mangling for section ids.
    assert result["id"] == "src-sec-rt::src/utils.py"
    assert result["mangled"] is False
    assert result["upserted"] is False  # first write, didn't already exist
    # File path: `::` → `/`, then `.md` appended.
    assert result["path"] == "pages/sources/src-sec-rt/src/utils.py.md"
    # Round-trip via read_page.
    read = _call_tool(
        mcp_client,
        sid,
        "read_page",
        {"page_id": "src-sec-rt::src/utils.py"},
        req_id=351,
    )
    assert read.get("error") is None
    assert read["id"] == "src-sec-rt::src/utils.py"
    assert read["title"] == "utils.py section"
    assert read["body"] == "section body"


def test_section_id_second_write_upserts(mcp_client: TestClient):
    """A second write_page with the same section id overwrites in place
    (upsert semantics: the id IS the canonical address)."""
    sid = _initialize(mcp_client)
    first = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section("src-sec-upsert::main.py", "v1"),
            "body": "first body",
        },
        req_id=352,
    )
    assert first["upserted"] is False
    second = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section("src-sec-upsert::main.py", "v2"),
            "body": "second body",
        },
        req_id=353,
    )
    assert second["upserted"] is True
    assert second["id"] == first["id"]  # canonical id stable
    assert second["path"] == first["path"]
    # Read back — body and title reflect the second write.
    read = _call_tool(
        mcp_client,
        sid,
        "read_page",
        {"page_id": "src-sec-upsert::main.py"},
        req_id=354,
    )
    assert read["title"] == "v2"
    assert read["body"] == "second body"


def test_section_id_index_special_case(mcp_client: TestClient):
    """`<src>::index` is the natural way to write the multi-file source's
    root index page → file at `pages/sources/<src>/index.md`."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section(
                "src-multi-index::index", "Multi-file source root"
            ),
            "body": "index body — points at sections",
        },
        req_id=355,
    )
    assert result.get("error") is None, result
    assert result["path"] == "pages/sources/src-multi-index/index.md"


def test_section_id_nested_subdirs(mcp_client: TestClient):
    """Deeply nested rel-paths are honored — multi-component paths
    translate to multi-level filesystem directories."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section(
                "src-deep::a/b/c/d/file.py", "deep section"
            ),
            "body": "",
        },
        req_id=356,
    )
    assert result["path"] == "pages/sources/src-deep/a/b/c/d/file.py.md"


def test_section_id_rejects_path_traversal(mcp_client: TestClient):
    """`..` in the rel-path is rejected at the schema validator
    (before reaching the filesystem)."""
    sid = _initialize(mcp_client)
    bad_ids = [
        "src-foo::../escape",
        "src-foo::a/../b",
        "src-foo::./hidden",
        "src-foo::../../top",
    ]
    for i, bad in enumerate(bad_ids):
        result = _call_tool(
            mcp_client,
            sid,
            "write_page",
            {"frontmatter": _src_fm_section(bad)},
            req_id=357 + i,
        )
        assert result["error"] == "validation_error", f"{bad!r} should reject"


def test_section_id_rejects_absolute_rel_path(mcp_client: TestClient):
    """Rel-path starting with `/` is an absolute path → rejected."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _src_fm_section("src-foo::/abs/path.py")},
        req_id=370,
    )
    assert result["error"] == "validation_error"
    assert "/" in result["message"] or "absolute" in result["message"].lower()


def test_section_id_rejects_empty_rel_path(mcp_client: TestClient):
    """Section id with empty rel-path (just `<src>::`) is rejected."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _src_fm_section("src-foo::")},
        req_id=371,
    )
    assert result["error"] == "validation_error"
    assert "rel-path" in result["message"] or "non-empty" in result["message"].lower()


def test_section_id_rejects_double_slash(mcp_client: TestClient):
    """`//` in rel-path (empty component) is rejected."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _src_fm_section("src-foo::a//b")},
        req_id=372,
    )
    assert result["error"] == "validation_error"


def test_section_id_rejects_multiple_separators(mcp_client: TestClient):
    """Section id with more than one `::` is rejected (ambiguous parse)."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _src_fm_section("src-foo::a::b")},
        req_id=373,
    )
    assert result["error"] == "validation_error"
    assert "::" in result["message"]


def test_section_id_rejects_windows_reserved_component(mcp_client: TestClient):
    """A rel-path component whose base (pre-extension) is a Windows-reserved
    name (CON, NUL, COM1, ...) is rejected case-insensitively."""
    sid = _initialize(mcp_client)
    bad_ids = ["src-foo::con.py", "src-foo::sub/nul.txt", "src-foo::COM1.h"]
    for i, bad in enumerate(bad_ids):
        result = _call_tool(
            mcp_client,
            sid,
            "write_page",
            {"frontmatter": _src_fm_section(bad)},
            req_id=380 + i,
        )
        assert result["error"] == "validation_error", f"{bad!r} should reject"


def test_section_id_rejects_hidden_file_component(mcp_client: TestClient):
    """A rel-path component starting with `.` (hidden file convention) is
    rejected — the leading-alphanumeric rule. Stops `.git/config` style
    paths from accidentally being treated as section ids."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _src_fm_section("src-foo::.hidden")},
        req_id=390,
    )
    assert result["error"] == "validation_error"


def test_section_id_with_dot_in_filename_ok(mcp_client: TestClient):
    """Dots in the middle of a filename are fine (`utils.py`, `package.json`,
    `data.test.json`); only LEADING dots are rejected."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": _src_fm_section(
                "src-dot-mid::pkg/data.test.json", "multi-dot filename"
            ),
            "body": "",
        },
        req_id=391,
    )
    assert result.get("error") is None
    assert result["path"] == "pages/sources/src-dot-mid/pkg/data.test.json.md"


def test_section_id_in_batch_write_pages(mcp_client: TestClient):
    """write_pages handles section-id entries with upsert semantics
    alongside slug-id (mangled) entries in the same batch."""
    sid = _initialize(mcp_client)
    pages_arg = [
        {
            "frontmatter": _ent_fm("ent-batch-with-section", "slug entry"),
            "body": "slug",
        },
        {
            "frontmatter": _src_fm_section(
                "src-batch-sec::file.py", "section entry"
            ),
            "body": "section",
        },
    ]
    result = _call_tool(
        mcp_client, sid, "write_pages", {"pages": pages_arg}, req_id=392
    )
    assert result.get("error") is None, result
    assert result["count"] == 2
    by_id = {w.get("original_id") or w["id"]: w for w in result["written"]}
    # Slug-id entry was mangled.
    slug_entry = by_id["ent-batch-with-section"]
    assert slug_entry["mangled"] is True
    # Section-id entry was NOT mangled; upserted=False (first write).
    section_entry = by_id["src-batch-sec::file.py"]
    assert section_entry["mangled"] is False
    assert section_entry["upserted"] is False
    assert section_entry["path"] == "pages/sources/src-batch-sec/file.py.md"


# ---------------------------------------------------------------------------
# write_page mode: create (always-mangle) / update (canonical id required)


def test_write_page_default_mode_is_create_and_mangles(mcp_client: TestClient):
    """Omitting `mode` defaults to `create` and produces a mangled canonical id."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-mode-default")
    result = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=440)
    assert result.get("error") is None
    assert result["mode"] == "create"
    assert result["mangled"] is True
    assert result["original_id"] == "ent-mode-default"
    assert result["id"] != "ent-mode-default"


def test_write_page_create_preserves_caller_id_in_aliases(mcp_client: TestClient):
    """The caller's id is added to `aliases` even if it was already
    populated; existing aliases are kept."""
    sid = _initialize(mcp_client)
    fm = dict(_ent_fm("ent-alias-preserve"), aliases=["pre-existing-alias"])
    result = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=400)
    assert result.get("error") is None
    # Read back; both the original id and the pre-existing alias should be present.
    read = _call_tool(
        mcp_client, sid, "read_page", {"page_id": result["id"]}, req_id=401
    )
    aliases = read["frontmatter"].get("aliases", [])
    assert "ent-alias-preserve" in aliases
    assert "pre-existing-alias" in aliases


def test_write_page_canonical_id_re_validates(mcp_client: TestClient):
    """The mangled canonical id (caller-id + '-' + 22-char base64) must pass
    the same _PAGE_ID_RE used for caller-supplied ids — so the canonical id
    is structurally consistent."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-revalidate-test")
    result = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=402)
    canonical = result["id"]
    # 22 chars of URL-safe base64 = [A-Za-z0-9_-], all valid for our regex.
    # The whole id must conform.
    import re
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,253}", canonical)


def test_write_page_mode_update_requires_canonical_id(mcp_client: TestClient):
    """Update mode against the caller-id (not the canonical id) of a prior
    create fails with not_found — there is no alias resolution in `update`."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-update-caller-id", "v1")
    create = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=420)
    assert create.get("error") is None
    # Update by the caller's original id (NOT the canonical id) must fail.
    update = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": {**fm, "title": "v2"}, "body": "x", "mode": "update"},
        req_id=421,
    )
    assert update["error"] == "not_found"
    assert update["id"] == "ent-update-caller-id"


def test_write_page_mode_update_with_canonical_id_modifies_in_place(mcp_client: TestClient):
    """Update with the canonical id succeeds; the on-disk page is overwritten
    and no new page is created."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-update-canon", "v1")
    create = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=430)
    canonical = create["id"]
    # Update using the canonical id.
    update = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {
            "frontmatter": {**fm, "id": canonical, "title": "v2"},
            "body": "updated body",
            "mode": "update",
        },
        req_id=431,
    )
    assert update.get("error") is None, update
    assert update["mode"] == "update"
    assert update["mangled"] is False
    assert update["id"] == canonical
    # Read back: changes are present.
    read = _call_tool(mcp_client, sid, "read_page", {"page_id": canonical}, req_id=432)
    assert read["title"] == "v2"
    assert read["body"] == "updated body"


def test_write_page_mode_update_fails_for_missing_canonical_id(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    # Construct a plausible-but-nonexistent canonical id.
    fake_canonical = "ent-not-there-AbCdEfGhIjKlMnOpQrStUv"
    fm = _ent_fm(fake_canonical)
    result = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": fm, "mode": "update"},
        req_id=433,
    )
    assert result["error"] == "not_found"
    assert result["id"] == fake_canonical


def test_write_page_mode_upsert_is_rejected(mcp_client: TestClient):
    """`upsert` is no longer a valid mode under always-mangle semantics."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-no-upsert")
    text = _call_tool_raw_text(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": fm, "mode": "upsert"},
        req_id=434,
    )
    # MCP's input schema enforces the enum (only create/update now).
    assert "mode" in text.lower() or "upsert" in text.lower()


# ---------------------------------------------------------------------------
# write_pages batch with mode


def test_write_pages_batch_create_mangles_each_entry(mcp_client: TestClient):
    """Every entry in a create batch gets its own mangled canonical id; same
    caller-id used twice produces two distinct canonical ids."""
    sid = _initialize(mcp_client)
    pages = [
        {"frontmatter": _ent_fm("ent-batch-mangle-twin", "A"), "body": "a"},
        {"frontmatter": _ent_fm("ent-batch-mangle-twin", "B"), "body": "b"},
    ]
    result = _call_tool(
        mcp_client,
        sid,
        "write_pages",
        {"pages": pages, "mode": "create"},
        req_id=450,
    )
    assert result.get("error") is None, result
    assert result["count"] == 2
    canonical_ids = {w["id"] for w in result["written"]}
    assert len(canonical_ids) == 2, "two creates with same caller id must produce distinct canonical ids"
    for w in result["written"]:
        assert w["mangled"] is True
        assert w["original_id"] == "ent-batch-mangle-twin"


def test_write_pages_batch_update_aborts_on_missing_id(mcp_client: TestClient):
    """Update mode requires every entry's id to be a valid existing
    canonical id; missing → entire batch aborts."""
    sid = _initialize(mcp_client)
    # Seed one canonical id first.
    seed = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-batch-update-seed", "preset")},
        req_id=470,
    )
    canonical = seed["id"]
    pages = [
        {"frontmatter": _ent_fm(canonical, "new title")},   # exists
        {"frontmatter": _ent_fm("ent-batch-update-nope-AbCdEfGhIjKlMnOpQrStUv")},  # doesn't exist
    ]
    result = _call_tool(
        mcp_client,
        sid,
        "write_pages",
        {"pages": pages, "mode": "update"},
        req_id=471,
    )
    assert result["error"] == "not_found"
    assert result["index"] == 1
    # The existing entry must NOT have been modified (validate-all-then-act).
    r = _call_tool(mcp_client, sid, "read_page", {"page_id": canonical}, req_id=472)
    assert r["title"] == "preset"


def test_write_pages_batch_create_default_mode(mcp_client: TestClient):
    """Default mode is `create`; every entry gets mangled by default."""
    sid = _initialize(mcp_client)
    pages = [
        {"frontmatter": _ent_fm("ent-batch-default-1", "A")},
        {"frontmatter": _ent_fm("ent-batch-default-2", "B")},
    ]
    result = _call_tool(
        mcp_client, sid, "write_pages", {"pages": pages}, req_id=480
    )
    assert result.get("error") is None, result
    assert result["mode"] == "create"
    assert result["count"] == 2
    assert all(w["mangled"] is True for w in result["written"])


# ---------------------------------------------------------------------------
# find_by_alias + read_page alias fallback


def test_find_by_alias_finds_post_mangle_caller_id(mcp_client: TestClient):
    """After write_page creates a mangled page, find_by_alias on the caller's
    original id returns the canonical match."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-alias-find-test")
    create = _call_tool(
        mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=500
    )
    canonical = create["id"]
    # Look up by caller's original id (which is in aliases).
    result = _call_tool(
        mcp_client,
        sid,
        "find_by_alias",
        {"alias": "ent-alias-find-test"},
        req_id=501,
    )
    assert result["count"] >= 1
    canonical_ids = {m["id"] for m in result["matches"]}
    assert canonical in canonical_ids
    # The matched row has the expected shape.
    for m in result["matches"]:
        assert {"id", "title", "type", "path"} <= m.keys()


def test_find_by_alias_returns_empty_for_unknown(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "find_by_alias",
        {"alias": "no-such-alias-anywhere"},
        req_id=502,
    )
    assert result["count"] == 0
    assert result["matches"] == []


def test_find_by_alias_returns_all_matches_for_collision(mcp_client: TestClient):
    """Two creates with the same caller id produce two pages sharing one
    alias; find_by_alias returns both."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-alias-twin", "A")
    a = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=503)
    b = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": {**fm, "title": "B"}},
        req_id=504,
    )
    result = _call_tool(
        mcp_client, sid, "find_by_alias", {"alias": "ent-alias-twin"}, req_id=505
    )
    assert result["count"] >= 2
    ids = {m["id"] for m in result["matches"]}
    assert a["id"] in ids
    assert b["id"] in ids


def test_read_page_falls_back_to_alias_when_single_match(mcp_client: TestClient):
    """A page created via write_page has a mangled canonical id; calling
    read_page with the caller's id (now an alias) resolves to that page
    and the response includes `resolved_via_alias`."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-alias-readback", "Read via alias")
    create = _call_tool(
        mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=510
    )
    canonical = create["id"]
    # Call read_page with the caller's original id (not the canonical).
    read = _call_tool(
        mcp_client,
        sid,
        "read_page",
        {"page_id": "ent-alias-readback"},
        req_id=511,
    )
    assert read.get("error") is None, read
    assert read["id"] == canonical
    assert read["title"] == "Read via alias"
    assert read.get("resolved_via_alias") == "ent-alias-readback"


def test_read_page_exact_id_wins_over_alias(mcp_client: TestClient):
    """If the page_id is itself a canonical id, exact match returns first
    (no `resolved_via_alias` field)."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-alias-exact-wins")
    create = _call_tool(
        mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=520
    )
    canonical = create["id"]
    # Exact-id read: no alias resolution should happen.
    read = _call_tool(
        mcp_client, sid, "read_page", {"page_id": canonical}, req_id=521
    )
    assert read["id"] == canonical
    assert "resolved_via_alias" not in read


def test_read_page_ambiguous_alias_returns_match_list(mcp_client: TestClient):
    """When the page_id matches multiple pages' aliases, read_page errors
    with {error: 'ambiguous_alias', matches: [...]} so the caller can pick
    one by canonical id."""
    sid = _initialize(mcp_client)
    fm = _ent_fm("ent-alias-ambig", "First")
    a = _call_tool(mcp_client, sid, "write_page", {"frontmatter": fm}, req_id=530)
    b = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": {**fm, "title": "Second"}},
        req_id=531,
    )
    # Now `ent-alias-ambig` is an alias on two distinct canonical ids.
    read = _call_tool(
        mcp_client,
        sid,
        "read_page",
        {"page_id": "ent-alias-ambig"},
        req_id=532,
    )
    assert read["error"] == "ambiguous_alias"
    assert read["alias"] == "ent-alias-ambig"
    match_ids = {m["id"] for m in read["matches"]}
    assert a["id"] in match_ids
    assert b["id"] in match_ids


def test_read_page_unknown_id_or_alias_still_returns_not_found(mcp_client: TestClient):
    """Truly missing id/alias still returns the existing not_found error."""
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client, sid, "read_page", {"page_id": "ent-truly-missing"}, req_id=540
    )
    assert result["error"] == "not_found"
    assert result["page_id"] == "ent-truly-missing"


# ---------------------------------------------------------------------------
# Scope-tier filtering (unit-level — exercising the helpers directly)


def test_scope_tier_filtering_read_only_hides_higher_tiers():
    """READ_ONLY scope sees only read_only tools — no write or remove ones."""
    from smalt_mcp.tools import list_tools
    from smalt_mcp.permissions import Scope

    names = {t.name for t in list_tools(Scope.READ_ONLY)}
    assert "status" in names
    assert "read_page" in names
    # READ_WRITE tools must be excluded
    assert "write_page" not in names
    assert "add_link" not in names
    # REMOVE_DESTRUCTIVE tools must be excluded
    assert "remove_page" not in names
    assert "update_claim" not in names


def test_scope_tier_filtering_read_write_hides_remove():
    """READ_WRITE scope sees read_only + read_write — no remove tools."""
    from smalt_mcp.tools import list_tools
    from smalt_mcp.permissions import Scope

    names = {t.name for t in list_tools(Scope.READ_WRITE)}
    assert "status" in names      # read_only
    assert "write_page" in names  # read_write
    assert "remove_page" not in names
    assert "update_claim" not in names
    assert "remove_claim" not in names
    assert "remove_link" not in names


def test_scope_tier_filtering_remove_destructive_sees_all():
    from smalt_mcp.tools import list_tools
    from smalt_mcp.permissions import Scope

    names = {t.name for t in list_tools(Scope.REMOVE_DESTRUCTIVE)}
    assert "status" in names
    assert "write_page" in names
    assert "remove_page" in names
    assert "update_claim" in names
    assert "remove_claim" in names
    assert "remove_link" in names


# ---------------------------------------------------------------------------
# incoming_links


def test_incoming_links_finds_what_links_to_seed_page(mcp_client: TestClient):
    """The seed Smalt has con-embedding → ent-alice (label: example_of), and
    con-index → con-embedding (label: built_over). incoming_links on each
    target should find them."""
    sid = _initialize(mcp_client)
    # Pages that link TO ent-alice
    result = _call_tool(
        mcp_client, sid, "incoming_links", {"page_id": "ent-alice"}, req_id=600
    )
    from_ids = {e["from_id"] for e in result["edges"]}
    assert "con-embedding" in from_ids
    # Pages that link TO con-embedding
    result2 = _call_tool(
        mcp_client, sid, "incoming_links", {"page_id": "con-embedding"}, req_id=601
    )
    from_ids2 = {e["from_id"] for e in result2["edges"]}
    assert "con-index" in from_ids2


def test_incoming_links_filter_by_label(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "incoming_links",
        {"page_id": "ent-alice", "label": "example_of"},
        req_id=602,
    )
    assert all(e["label"] == "example_of" for e in result["edges"])
    # Wrong label → empty
    empty = _call_tool(
        mcp_client,
        sid,
        "incoming_links",
        {"page_id": "ent-alice", "label": "no_such_label"},
        req_id=603,
    )
    assert empty["count"] == 0


def test_incoming_links_unknown_page_returns_empty(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "incoming_links",
        {"page_id": "ent-nobody-points-here"},
        req_id=604,
    )
    assert result["count"] == 0
    assert result["edges"] == []


# ---------------------------------------------------------------------------
# remove_page (cascading)


def test_remove_page_cascades_links_and_claims(mcp_client: TestClient):
    """Create a page, link to it from another, add a claim — then remove_page
    cleans up everything."""
    sid = _initialize(mcp_client)
    # Create the page that will be removed.
    target = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-remove-target", "target")},
        req_id=700,
    )
    target_id = target["id"]
    # Create another page that links TO it.
    referrer = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-remove-referrer", "referrer")},
        req_id=701,
    )
    referrer_id = referrer["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": referrer_id, "to_id": target_id, "label": "knows"},
        req_id=702,
    )
    # Add a claim on the target.
    _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {
            "page_id": target_id,
            "claim": {"id": "claim-on-target-1", "text": "fact"},
        },
        req_id=703,
    )
    # Pre-check: incoming_links sees the inbound edge; read_page works.
    inc = _call_tool(mcp_client, sid, "incoming_links", {"page_id": target_id}, req_id=704)
    assert any(e["from_id"] == referrer_id for e in inc["edges"])

    # Now remove
    result = _call_tool(
        mcp_client, sid, "remove_page", {"page_id": target_id}, req_id=705
    )
    assert result["id"] == target_id
    assert result["removed"]["embedding"] == 1
    assert result["removed"]["incoming_links"] >= 1
    assert result["removed"]["claims"] >= 1

    # Verify: read_page now returns not_found (no exact match, no alias either).
    r = _call_tool(mcp_client, sid, "read_page", {"page_id": target_id}, req_id=706)
    assert r["error"] == "not_found"
    # The referrer's outgoing link is gone from the index too.
    trav = _call_tool(
        mcp_client,
        sid,
        "traverse",
        {"from_id": referrer_id, "label": "knows"},
        req_id=707,
    )
    target_ids = {e["to_id"] for e in trav["edges"]}
    assert target_id not in target_ids


def test_remove_page_not_found(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "remove_page",
        {"page_id": "ent-nope-AbCdEfGhIjKlMnOpQrStUv"},
        req_id=710,
    )
    assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# update_claim


def test_update_claim_replaces_in_place(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    # Set up: a page with a claim.
    page = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-update-claim", "host")},
        req_id=720,
    )
    page_id = page["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {
            "page_id": page_id,
            "claim": {"id": "c-1", "text": "v1", "confidence": 0.3},
        },
        req_id=721,
    )
    # Update it.
    result = _call_tool(
        mcp_client,
        sid,
        "update_claim",
        {
            "page_id": page_id,
            "claim_id": "c-1",
            "new_claim": {"id": "c-1", "text": "v2", "confidence": 0.9},
        },
        req_id=722,
    )
    assert result["updated"] is True
    # Read back: the claim text + confidence changed.
    read = _call_tool(mcp_client, sid, "read_page", {"page_id": page_id}, req_id=723)
    claims = read["frontmatter"].get("claims", [])
    c = next(c for c in claims if c.get("id") == "c-1")
    assert c["text"] == "v2"
    assert c["confidence"] == 0.9


def test_update_claim_claim_not_found(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    page = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-update-claim-missing", "host")},
        req_id=730,
    )
    page_id = page["id"]
    result = _call_tool(
        mcp_client,
        sid,
        "update_claim",
        {
            "page_id": page_id,
            "claim_id": "no-such-claim",
            "new_claim": {"id": "no-such-claim", "text": "x"},
        },
        req_id=731,
    )
    assert result["error"] == "claim_not_found"


def test_update_claim_id_mismatch_rejected(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    page = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-update-mismatch", "host")},
        req_id=740,
    )
    page_id = page["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": page_id, "claim": {"id": "c-orig", "text": "x"}},
        req_id=741,
    )
    # new_claim.id != claim_id → rejected
    result = _call_tool(
        mcp_client,
        sid,
        "update_claim",
        {
            "page_id": page_id,
            "claim_id": "c-orig",
            "new_claim": {"id": "c-renamed", "text": "x"},
        },
        req_id=742,
    )
    assert result["error"] == "invalid_argument"


# ---------------------------------------------------------------------------
# remove_claim


def test_remove_claim_removes(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    page = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-remove-claim", "host")},
        req_id=750,
    )
    page_id = page["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": page_id, "claim": {"id": "c-keep", "text": "keep me"}},
        req_id=751,
    )
    _call_tool(
        mcp_client,
        sid,
        "add_claim",
        {"page_id": page_id, "claim": {"id": "c-drop", "text": "drop me"}},
        req_id=752,
    )
    result = _call_tool(
        mcp_client,
        sid,
        "remove_claim",
        {"page_id": page_id, "claim_id": "c-drop"},
        req_id=753,
    )
    assert result["removed"] is True
    # Read back: c-keep present, c-drop absent.
    read = _call_tool(mcp_client, sid, "read_page", {"page_id": page_id}, req_id=754)
    ids = {c.get("id") for c in read["frontmatter"].get("claims", [])}
    assert "c-keep" in ids
    assert "c-drop" not in ids


def test_remove_claim_not_found(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    page = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-remove-claim-miss", "host")},
        req_id=760,
    )
    result = _call_tool(
        mcp_client,
        sid,
        "remove_claim",
        {"page_id": page["id"], "claim_id": "no-such-claim"},
        req_id=761,
    )
    assert result["error"] == "claim_not_found"


# ---------------------------------------------------------------------------
# remove_link


def test_remove_link_with_label_removes_matching_edge(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    # Set up: two pages, two distinct labeled edges between them.
    src = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-rmlink-src", "src")},
        req_id=770,
    )
    dst = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-rmlink-dst", "dst")},
        req_id=771,
    )
    src_id, dst_id = src["id"], dst["id"]
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": src_id, "to_id": dst_id, "label": "knows"},
        req_id=772,
    )
    _call_tool(
        mcp_client,
        sid,
        "add_link",
        {"from_id": src_id, "to_id": dst_id, "label": "cites"},
        req_id=773,
    )
    # Remove only the `knows` edge.
    result = _call_tool(
        mcp_client,
        sid,
        "remove_link",
        {"from_id": src_id, "to_id": dst_id, "label": "knows"},
        req_id=774,
    )
    assert result["removed"] == 1
    # `cites` survives.
    trav = _call_tool(
        mcp_client, sid, "traverse", {"from_id": src_id}, req_id=775
    )
    labels = {e["label"] for e in trav["edges"] if e["to_id"] == dst_id}
    assert "cites" in labels
    assert "knows" not in labels


def test_remove_link_without_label_removes_all_between_pair(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    src = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-rmlink-allsrc", "src")},
        req_id=780,
    )
    dst = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-rmlink-alldst", "dst")},
        req_id=781,
    )
    src_id, dst_id = src["id"], dst["id"]
    for lbl in ("a", "b", "c"):
        _call_tool(
            mcp_client,
            sid,
            "add_link",
            {"from_id": src_id, "to_id": dst_id, "label": lbl},
            req_id=782,
        )
    # Remove ALL edges from src → dst regardless of label.
    result = _call_tool(
        mcp_client,
        sid,
        "remove_link",
        {"from_id": src_id, "to_id": dst_id},
        req_id=783,
    )
    assert result["removed"] == 3
    trav = _call_tool(mcp_client, sid, "traverse", {"from_id": src_id}, req_id=784)
    assert all(e["to_id"] != dst_id for e in trav["edges"])


def test_remove_link_no_match_returns_zero(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    src = _call_tool(
        mcp_client,
        sid,
        "write_page",
        {"frontmatter": _ent_fm("ent-rmlink-nomatch", "src")},
        req_id=790,
    )
    result = _call_tool(
        mcp_client,
        sid,
        "remove_link",
        {"from_id": src["id"], "to_id": "ent-never-linked", "label": "x"},
        req_id=791,
    )
    assert result["removed"] == 0
    assert result.get("reason") == "no_matching_link"


def test_remove_link_unknown_from_page(mcp_client: TestClient):
    sid = _initialize(mcp_client)
    result = _call_tool(
        mcp_client,
        sid,
        "remove_link",
        {
            "from_id": "ent-no-such-page-AbCdEfGhIjKlMnOpQrStUv",
            "to_id": "ent-target",
        },
        req_id=795,
    )
    assert result["error"] == "not_found"
