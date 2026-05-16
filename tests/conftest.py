"""Shared test fixtures.

The MCP `/sse` transport relies on a `StreamableHTTPSessionManager` that
hard-errors on `run()` being called twice. Each `with TestClient(app) as c`
block runs the FastAPI lifespan, which calls `session_manager.run()`. So we
can't have two MCP-using test modules each owning their own `with`-scoped
client — the second module's lifespan startup blows up.

The fix: a single session-scoped TestClient lives here; every test module
that touches `/sse` reuses it.

We also bootstrap a tiny seed Smalt at the start of the session (via the
indexer) so read-only tools have something to read. The seed is built once
per test session; tests rely on its shape.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Seed Smalt — small, deterministic, covers all four page types and a chain
# of links so traverse() has something to walk.

_SEED_PAGES: dict[str, str] = {
    "pages/entities/alice.md": """---
id: ent-alice
type: entity
title: Alice
aliases: [Alicia]
tags: [person]
entity_kind: person
domains: [con-cs]
---
Alice is a fictional person used in the seed Smalt for testing.
""",
    "pages/entities/bob.md": """---
id: ent-bob
type: entity
title: Bob
tags: [person]
entity_kind: person
---
Bob is another fictional person in the seed Smalt.
""",
    "pages/concepts/cs.md": """---
id: con-cs
type: concept
title: Computer Science
tags: [domain]
is_domain: true
---
Computer Science — a seed domain ConceptPage for testing list_domains.
""",
    "pages/concepts/embedding.md": """---
id: con-embedding
type: concept
title: Embedding
tags: [ml]
parents: []
domains: [con-cs]
glossary: true
links_out:
  - target: ent-alice
    label: example_of
evidence:
  - source_id: src-doc1
    snippet: "embedding research"
---
An embedding is a dense vector representation of structured data.
Alice frequently appears in embedding research.
""",
    "pages/concepts/index.md": """---
id: con-index
type: concept
title: Index
tags: [database, search]
parents: []
domains: [con-cs]
glossary: true
links_out:
  - target: con-embedding
    label: built_over
---
An index is a derived structure that accelerates queries.
Vector indexes are typically built over embeddings.
""",
    "pages/sources/doc1.md": """---
id: src-doc1
type: source
title: Seed Doc 1
tags: [seed]
location_uri: file:/tmp/seed/doc1.md
location_kind: file
domains: [con-cs]
---
This is a seed source page. It mentions Alice and embedding by name so
that search has something to match against.
""",
}


def _write_seed_smalt(smalt_root: Path) -> None:
    """Materialize the seed pages into `smalt_root/pages/<...>.md`."""
    for rel, content in _SEED_PAGES.items():
        target = smalt_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _bootstrap_and_index(smalt_root: Path) -> None:
    """Create LanceDB tables and run the indexer over the seed pages.

    Uses the `fake` embedder (set via EMBEDDING_PROVIDER=fake before this
    function is called) so the test doesn't pay a fastembed model-download
    cost.
    """
    from smalt_mcp.config import load_config
    from smalt_mcp.storage.embedder import make_embedder
    from smalt_mcp.storage.indexer import Indexer
    from smalt_mcp.storage.lance import connect, ensure_tables

    cfg = load_config()
    ensure_tables(smalt_root, embedding_dim=cfg.embedding.dim)
    embedder = make_embedder(cfg)
    db = connect(smalt_root)
    Indexer(smalt_root=smalt_root, embedder=embedder, db=db).run()


@pytest.fixture(scope="session")
def mcp_client(tmp_path_factory) -> TestClient:
    """One TestClient for the whole test session — drives the MCP `/sse`
    surface, lifespan started exactly once. Bootstraps a seeded Smalt before
    importing the server module so the App picks up the configured path.
    """

    smalt_dir = tmp_path_factory.mktemp("smalt")

    # Env var trifecta must be set BEFORE the first `from smalt_mcp.server`
    # import, because server.py calls `App()` at module load and that reads
    # the env via load_config().
    os.environ["SMALT_DIR"] = str(smalt_dir)
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    os.environ["EMBEDDING_DIM"] = "384"

    _write_seed_smalt(smalt_dir)
    _bootstrap_and_index(smalt_dir)

    from smalt_mcp.server import app

    with TestClient(app) as c:
        yield c
