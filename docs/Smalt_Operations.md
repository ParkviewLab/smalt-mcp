<!--
SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Smalt Operations

What `smalt-mcp` does, and how it does it. This document covers the concept, the place the
server occupies in the wider CoGrind system, the data model, the retrieval machinery, the
full MCP tool surface, and the operational surface (running, configuring, backing up, and
the ways it can fail). It is written against the code rather than against the prose; see
[Document currency](#document-currency) at the end.

## What the Smalt is

The server's own one-line definition, from `README.md` and `pyproject.toml`:

> MCP server wrapping the **Smalt**'s storage surface (read / write / link / claim /
> search) for ParkviewLab's CoGrind project. Thinnest viable wrapper around markdown +
> LanceDB; no agentic logic. Single-writer to a given Smalt.

A *Smalt* is a knowledge corpus: a directory of markdown pages with YAML frontmatter,
describing entities, concepts, sources, and syntheses, joined by labelled links and
annotated with claims. `smalt-mcp` is the MCP server that reads and writes that corpus and
serves retrieval over it.

The framing axiom is stated in `README.md:11`: this server is "to `cobalt-grinding` what
`deco-assaying` is to tree-sitter: a clean MCP-shaped wrapper around a deterministic
capability." The emphasis belongs on *deterministic*. If one arrives suspecting the project
is "about knowledge semantics," that is right in substance: it maintains a typed knowledge
graph, embeds page bodies as dense vectors, and answers queries by fusing semantic,
lexical, and alias retrieval. But it is the storage and retrieval layer of a knowledge
system, not a reasoning engine. There is no agentic logic in it by design; every judgment
call lives in the orchestrator that consumes it.

Two properties follow from that stance and are worth holding onto, because most of the
design falls out of them:

Markdown is canonical, and the index is derived. The pages on disk are the truth; the
LanceDB store is a rebuildable projection of them. One may delete the entire index and
reconstruct it from `pages/`.

There is exactly one writer to a given Smalt. Concurrency inside the process is real
(handlers run on a thread pool), but the commit phase of every write serializes through a
single mutex.

The name follows the family's pigment convention: grinding fired cobalt-blue glass yields a
fine blue pigment called smalt. CoGrind grinds cobalt to make Smalt; the Smalt is what the
system knows.

## Where it sits: the CoGrind system

`smalt-mcp` is not a standalone product. It is one of two memory substrates belonging to
**cobalt-grinding** (CoGrind), which its own northstar describes as "an MCP server wrapped
around an AI brain." Understanding the boundary is most of understanding this server,
because so much of what one might expect to find here deliberately lives elsewhere.

### The brain and its two memories

CoGrind's memory is split across two substrates, each served by a separate MCP server that
CoGrind supervises as a child process:

| Substrate | Server | Holds | Storage |
|---|---|---|---|
| The Smalt | `smalt-mcp` | Canonical knowledge: what the system knows | Markdown + LanceDB index |
| The lab notebook | `ebony-enriching` | Research in flight: proposals, experiments, gaps | Markdown only, no index |

The distinction CoGrind's northstar draws is that the lab notebook is what the system is
*thinking about*, while the Smalt is what it *knows*. Knowledge is promoted from the one to
the other only after it survives a test.

### The topology

The shape is a supervised star, not a linear pipeline. CoGrind is simultaneously an MCP
server (exposing `wiki.*` tools to clients such as Claude Desktop or the `cogrind-workshop`
CLI) and an MCP host (spawning and calling children on behalf of its own agents):

```
                        ┌──────────────────┐
   MCP clients ────────►│  cobalt-grinding │  the brain: ingest, retrieve,
   (Claude, workshop)   │  (MCP server +   │  converse, gap analysis
                        │   MCP host)      │
                        └────────┬─────────┘
                                 │ MCP (as host)
             ┌───────────────┬───┴───────────┬────────────────┐
             ▼               ▼               ▼                ▼
      ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │ smalt-mcp  │  │   ebony-     │  │    deco-     │  │    flint-    │
      │  :35833    │  │  enriching   │  │  assaying    │  │   slating    │
      │            │  │   :35834     │  │   :35832     │  │              │
      │ the Smalt  │  │ lab notebook │  │ tree-sitter  │  │  PDF reading │
      └────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

The ports are allocated sequentially across the family: `deco-assaying` on 35832,
`smalt-mcp` on 35833, `ebony-enriching` on 35834.

The rule that matters most: **the two substrates have zero outbound dependencies, and
neither knows the other exists.** `smalt-mcp` imports nothing from `ebony-enriching`, calls
none of its tools, and shares no file or index with it. Every cross-substrate operation is
orchestrated by CoGrind, in sequence, from the outside. The canonical example is applying a
validated proposal, which is `smalt.write_page` followed by
`ebony.update_proposal_status(applied)`; two calls, two children, one orchestrator.

This is why `smalt-mcp` can be understood, run, and tested entirely on its own, and why its
tests need no sibling running.

### What CoGrind writes into the Smalt

During ingestion, CoGrind points itself at a file, a directory, a git URL, or a PDF URL,
and grinds it down into pages that it writes here through `smalt.write_page` and
`smalt.add_links`. Broadly it produces a `SourcePage` per file (and a source-index page per
directory), section pages beneath it, `EntityPage`s for people, organizations, products,
repositories, and packages, glossary `ConceptPage`s carrying a term and its evidence, and
labelled edges between them (`mentioned_in`, `defined_in`, `mentions`, `defines`,
`contains`).

One property of that ingestion governs what a SourcePage means here: **CoGrind never copies
the source.** Its northstar puts it as "memory is earned, not stored." What lands in the
Smalt is the source's location, its content hash at fetch time, the fetch timestamp, its
structure, and the notes the system made about it. Re-verification re-fetches from the
recorded location, and re-ingestion short-circuits when the hash is unchanged. A SourcePage
is therefore a durable, checkable reference plus derived understanding, not an archive copy.

A related framing from the same northstar is worth quoting because it is easy to
misclassify this server: CoGrind is explicitly "not a RAG implementation." Retrieval runs
against interlinked structured notes that the system wrote, not against raw chunks of
source text. That the notes happen to be embedded and searched by vector similarity is an
implementation detail of finding them, not the substance of the design.

### What the Smalt is deliberately not

The scientific-method surface is not here. Proposals, experiments, and knowledge gaps live
in `ebony-enriching`. This was not an oversight; those tools existed in `smalt-mcp` and
were removed in v0.5.0 when the substrate split happened. The code says so in three places,
so that nobody re-adds them by accident. From `schema.py`:

> Proposals (the scientific-method surface) live in a separate substrate: the
> `ebony-enriching` MCP server, with its own `ProposalPage` schema and
> `EBONY_ENRICHING_DIR` storage. Smalt-mcp is purely the canonical knowledge substrate; no
> proposal / experiment / gap models live here.

The `tasks/` directory carries the same warning, and the `SCHEMA.md` and `POLICY.md`
placeholders written at bootstrap state that schema and policy changes are proposed,
tested, and applied through `ebony-enriching`.

### The flywheel that joins the three

The three repos connect through a loop that CoGrind drives end to end. Retrieval, ingestion,
and conversation notice what the corpus cannot answer and emit a gap (`ebony.add_gap`).
Research reads the accumulated gaps (`ebony.list_gaps`), and proposes an answer as a
falsifiable hypothesis (`ebony.write_proposal`). A human approves it. CoGrind then applies
it across both substrates: the knowledge lands in the Smalt (`smalt.write_page`) and the
proposal is marked `applied` in the notebook. Truth is provisional; the lifecycle is
Observe, Hypothesize, Predict, Test, Validate, Apply.

`smalt-mcp` participates in that loop only at the last step, and only as a place to write.
It has no opinion about proposals and never learns that one existed.

## The data model

Pages are Pydantic models in `schema.py`, resolved as a discriminated union on the `type`
field.

| Page type | `type` | What it represents |
|---|---|---|
| `EntityPage` | `entity` | A person, organization, product, repository, package |
| `ConceptPage` | `concept` | An idea or term; flagged `is_domain` for a domain, `glossary` for a glossary term |
| `SourcePage` | `source` | An ingested source: its location, hash, structure, and summary |
| `SynthesisPage` | `synthesis` | Knowledge composed across other pages |
| `IndexPage` | `index` | An auto-generated index (`glossary.md`, `domains.md`) |

Every page shares `PageBase`: an `id`, `type`, `title`, `aliases`, `tags`, `created_at` and
`updated_at`, and `links_out`. Two scoring fields, `quality_score` and `veracity_score`, are
reserved for future veracity work. `PageBase` sets `extra="allow"`, so a page carrying
frontmatter fields this version does not know about still parses; the schema is
forward-compatible on purpose.

Three building blocks compose onto pages:

`Link` is a labelled outgoing edge, `{target, label?, via_source?}`. Edges live on the page
that emits them, in `links_out`; the index derives the reverse direction, which is what
`incoming_links` reads.

`Claim` is a typed assertion attached to a page: `{id, text, value_type?, value?, unit?,
confidence?, confidence_label, source_ref?}`. The `source_ref` is a durable pointer, in
forms such as `git:<remote>@<sha>:<path>`.

`Evidence` is `{source_id, snippet}`, the "how do we know this" attached to a claim or a
glossary definition.

Identifiers are validated as path components, since an id becomes a filename. Traversal
sequences, slashes, and Windows-reserved names are rejected. Ids containing `::` are
special: they denote a section page beneath a source, and `::` maps to `/` on disk, so
`<source-id>::src/utils.py` becomes `pages/sources/<source-id>/src/utils.py.md`. Section ids
are also the one write path that upserts rather than mangling (see
[Write semantics](#write-semantics)).

Index pages are generated, not authored. The indexer regenerates `glossary.md` and
`domains.md` from current corpus state on every pass, sorted deterministically by id.
Writing one directly is rejected with `forbidden_page_type`; removing one is permitted.

## On-disk layout

A Smalt is a directory, rooted at `SMALT_DIR`. `bootstrap` creates this tree:

```
$SMALT_DIR/
├── pages/
│   ├── entities/       # EntityPage markdown
│   ├── concepts/       # ConceptPage markdown
│   ├── sources/        # SourcePage markdown; section pages nest beneath
│   ├── syntheses/      # SynthesisPage markdown
│   ├── glossary.md     # generated IndexPage
│   └── domains.md      # generated IndexPage
├── structures/         # sidecar JSON for oversized source structures
├── schema/
│   ├── SCHEMA.md       # living, human-facing schema doc
│   └── POLICY.md       # living, human-facing policy doc
├── index/
│   └── lance/          # LanceDB store: derived, rebuildable, excluded from backup
└── tasks/              # reserved for future Smalt-internal task state
```

The important line runs between `pages/` and `index/lance/`. Everything under `pages/` is
the source of truth and deserves backup and version control. Everything under `index/lance/`
is a projection that `reindex_all` can rebuild from scratch.

## How the retrieval works

This is the knowledge-semantics machinery proper: how a page becomes searchable, and how a
query finds it.

### Embeddings

Embedding is abstracted behind an `Embedder` protocol exposing `dim`, `model_version`
(formatted `<provider>:<model>`), and `embed(texts)`. `make_embedder(cfg)` selects the
implementation:

| Provider | Status | Notes |
|---|---|---|
| `fastembed` | Default, the only one wired | Local ONNX inference; default model `BAAI/bge-small-en-v1.5`, dim 384. No network call at query time, no API key. |
| `fake` | Test hook | Deterministic SHA-256-derived vectors of the same dim. Embedding quality is irrelevant; it exists so tests need not download a model. Set `EMBEDDING_PROVIDER=fake`. |
| `voyage`, `openai` | Placeholders | `make_embedder` raises `NotImplementedError`. |

That `fastembed` is local and default is a deliberate operational property: a Smalt indexes
and searches with no external dependency.

### The indexer: markdown to LanceDB

`Indexer.run()` walks `pages/`, parses and validates each file, and projects it into the
index. It is incremental by default: each page's `content_hash` (a SHA-256 of the file
bytes) is compared against the indexed row, and unchanged files are skipped without being
re-embedded, which matters because embedding is the expensive step. Rows whose files have
disappeared are swept. Changed bodies are embedded in a batch, and everything is written
through merge-insert upserts into the tables. It runs in three modes: incremental (the
default, and what fires after every write), full (`reindex_all`), and path-scoped
(`reindex_page`).

Two details are worth knowing because they bound retrieval quality:

**There is no real chunking yet.** Each page produces exactly one embedding, computed over
its body truncated to a fixed 2000-character budget (`EMBED_BODY_CHAR_BUDGET`). The module
docstring says plainly that "chunking is planned for a later iteration." The practical
consequence is that vector recall on a long page reflects only its opening; the coarse
substitute is granularity at the page level, which is why ingestion emits one section page
per file rather than one page per repository.

**Index pages are regenerated at the end of every pass**, so `glossary.md` and `domains.md`
are always consistent with the corpus that was just indexed.

### The LanceDB index

One LanceDB database lives under `index/lance/`, with five tables:

| Table | Contents |
|---|---|
| `pages` | `id, path, type, title, body, frontmatter_json, content_hash, created_at, updated_at, aliases` |
| `embeddings` | `page_id, vector (float32[dim]), model_version` |
| `links` | The edge set, projected from every page's `links_out` |
| `claims` | Claims, projected from pages |
| `sources` | Source records |

Two derived indexes sit on top:

Full-text search is LanceDB-native, built one field at a time on `title` and `body`, with
per-field status recorded for observability.

Vector search is an IVF-PQ approximate-nearest-neighbour index on `embeddings.vector` with
`metric="cosine"`. Below 256 embeddings it is intentionally skipped and the query falls back
to a brute-force scan. That is a normal, correct state reported as `skipped`, not a failure:
IVF-PQ needs enough rows to partition meaningfully, and a brute-force scan over a small
corpus is both exact and fast.

### Hybrid search

`search` retrieves from three independent sources and fuses them:

1. Full-text search over the `pages` table.
2. Vector search: the query is embedded, then matched by cosine similarity over `embeddings`.
3. Alias matching: pages whose `aliases` contain the whole query or any whitespace-delimited
   token of it.

The three ranked lists are combined by Reciprocal Rank Fusion with `k = 60`. RRF is the
right instrument here because the three sources produce incomparable scores; a BM25 score
and a cosine distance cannot be averaged meaningfully, but their *ranks* can be. Each hit
contributes `1/(k + rank)`, so a document ranked well by two sources beats one ranked
brilliantly by a single source. Property filters apply after fusion and before truncation,
and each hit returns `{id, aliases, title, type, snippet, score}`, the snippet being the
first 200 characters of the body.

Search degrades rather than fails: if full-text search is unavailable, it proceeds with
vector and alias retrieval.

Fuzzy alias resolution is a separate mechanism, and a common point of confusion. It is
trigram-set Jaccard similarity with a default threshold of 0.6, tunable through
`SMALT_FUZZY_ALIAS_THRESHOLD`, and it is used only by `read_page` and `find_by_alias` as a
fallback when an exact id or alias misses. It is **not** part of `search`.

`source_similarity` is the other retrieval path: given a source's id, it ranks pages by
cosine similarity against that source's already-stored embedding, which requires no new
embedding call.

### A request, end to end

```
MCP client
  └─ POST /sse                          Streamable HTTP transport, stateless
      └─ StreamableHTTPSessionManager
          └─ @mcp.call_tool  (server.py)
              └─ tools.dispatch(name, args, app, scope)
                  ├─ tier check          insufficient scope → PermissionError → "forbidden"
                  └─ tool.handler(app, args)
                      ├─ sync handlers run via asyncio.to_thread on the thread pool
                      ├─ reads:  app.db()  (lazy, lock-guarded)  /  app.embedder()
                      └─ writes: app.mutex.acquire(name)
                                   ├─ atomic write: tmp file, then os.replace
                                   └─ indexer pass
                          └─ result dict → one JSON TextContent block
```

Errors are mapped rather than escaping: an unknown name yields `unknown_tool`, insufficient
scope yields `forbidden`, and any other exception yields `tool_error`.

## The MCP tool surface

The server exposes 27 tools and no MCP resources or prompts. Tools are registered as frozen
`ToolDef(spec, scope, handler)` records in a single `TOOLS` registry, so adding one is a
registry entry plus a handler, with no edit to `server.py`.

Access is tiered by `SMALT_SCOPE`: a caller at tier N sees and may call every tool whose
required scope is at or below N. The tiers are `read_only` (0), `read_write` (1), and
`remove_destructive` (2).

### `read_only` (12 tools)

| Tool | What it does |
|---|---|
| `status` | Smalt path and existence, LanceDB table inventory, page count, mutex state, embedding provider/model/dim. |
| `index_status` | Deep health: per-table row counts, last index result, per-field FTS status, ANN status, embedding config, mutex contention. Same payload as `GET /admin/health`. |
| `list_pages` | Indexed pages as `{id, title, type, path}`; filters on `type`/`prefix` plus property filters; reports `truncated`. |
| `read_page` | A full page, frontmatter and body. Resolves by exact id, then exact alias, then fuzzy alias; reports `ambiguous_alias` on multiple matches. |
| `find_by_alias` | Every page whose `aliases` contains the given alias, exact then fuzzy. |
| `traverse` | Breadth-first walk of outgoing edges from a page; optional per-hop label filter; `hops` defaults to 1 and is capped at 5. |
| `incoming_links` | The inverse of `traverse`: every edge pointing at a page. Intended for auditing before a removal. |
| `search` | Hybrid FTS + vector + alias retrieval, RRF-fused. |
| `list_domains` | ConceptPages flagged `is_domain: true`. |
| `source_similarity` | Pages most similar to a source's stored embedding, by cosine similarity. |
| `task_status` | State of one async task. |
| `task_list` | Async tasks, filtered by state or kind. |

### `read_write` (11 tools)

| Tool | What it does |
|---|---|
| `bootstrap` | Create the canonical directories, write the `SCHEMA.md`/`POLICY.md` placeholders, create the LanceDB tables, run one indexer pass. Idempotent. |
| `write_page` | Create or update a page. See [Write semantics](#write-semantics). |
| `write_pages` | A batch of writes; validate-all-then-act, with a single indexer pass at the end. |
| `add_link` | Append an outgoing link to `links_out`; a duplicate `(target, label)` is a no-op. |
| `add_claim` | Append a validated claim; a duplicate id is a no-op. |
| `add_links` | Batch form of `add_link`, one read/write/index pass. |
| `add_claims` | Batch form of `add_claim`. |
| `write_batch` | A mixed-operation atomic transaction over page writes, links, and claims; three phases (validate, existence-check, commit). Cross-operation references within one batch are not supported. |
| `reindex_page` | Force one page to be re-indexed from disk. |
| `reindex_all` | Wipe and rebuild the whole index. Asynchronous: returns a `task_id` to poll. |
| `task_cancel` | Cooperatively cancel an async task. |

### `remove_destructive` (4 tools)

| Tool | What it does |
|---|---|
| `remove_page` | Cascading delete: the file, its `pages` row, its embedding, its outgoing links, every incoming link, and its claims. |
| `update_claim` | Replace one claim by id; the replacement's `id` must match. |
| `remove_claim` | Remove one claim by id. |
| `remove_link` | Remove edges by `(from_id, to_id, label?)`; omit the label to drop every edge between the pair. |

### Write semantics

Three behaviours are unobvious and worth stating explicitly.

**Create always mangles the id.** `write_page` in `create` mode does not honour the caller's
id verbatim; it appends a 22-character URL-safe base64 UUID4 suffix and preserves the
caller's original id in `aliases`. The collision probability is negligible, and the page
stays reachable by the memorable handle through alias resolution. There is deliberately no
`upsert` mode: create cannot silently clobber. `update` mode, by contrast, requires an
existing canonical id.

**Section ids are the exception.** An id containing `::` takes an upsert path instead of
being mangled, which is what gives re-ingestion a stable identity for the same file across
runs.

**Every write triggers the indexer.** This is what keeps the index consistent with the
markdown without a background daemon. It is also why the batch tools exist: `write_pages`,
`add_links`, `add_claims`, and `write_batch` collapse to a single indexer pass, amortizing
the embedding and LanceDB cost that would otherwise be paid per item.

## Runtime and operations

### Running it

The entry point is `python -m smalt_mcp`, or the `smalt-mcp` console script. Two transports
are available: HTTP (the default), serving uvicorn on `smalt_mcp.server:app`, and stdio via
`--transport stdio` for direct MCP-child use. The README documents five deployment modes,
matching the family pattern: `uvx` for a one-off, `uv tool install` for a pinned daemon on
`$PATH`, a macOS LaunchAgent or a Linux systemd user unit for a persistent daemon, and
Docker or docker compose for a container. The Docker image sets `SMALT_DIR=/data` and
exposes 35833; the compose file defaults to `SMALT_SCOPE=read_only`.

A first run against a fresh directory needs exactly one call to `bootstrap`.

### Configuration

All configuration is environment-driven; `config.py` is a pure leaf module with no internal
imports.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `35833` | HTTP listen port (`deco-assaying`'s 35832 plus one). |
| `HOST` | `0.0.0.0` | HTTP bind address. |
| `SMALT_DIR` | `~/Documents/Smalt` | The Smalt this server wraps. Expanded and resolved at startup. |
| `SMALT_SCOPE` | `read_write` | Permission tier: `read_only`, `read_write`, or `remove_destructive`. |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` is wired; `fake` is the test hook; `voyage`/`openai` are placeholders. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Model passed to the provider. |
| `EMBEDDING_DIM` | `384` | Must match the model. |
| `SMALT_THREAD_POOL_WORKERS` | `32` | Bounds concurrent handler execution on the loop's thread pool. |
| `SMALT_FUZZY_ALIAS_THRESHOLD` | `0.6` | Trigram-Jaccard threshold for fuzzy alias resolution. |
| `SMALT_INTERNAL_TOKEN` | unset | Reserved for future per-client scope routing. Not yet enforced. |

The scope is parsed once at startup, so the tier is a property of the running server, not of
the caller. Running two servers at different scopes against one Smalt is the way to give
different clients different authority, subject to the single-writer rule.

### HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `POST /sse` | The MCP Streamable HTTP transport (also accepts `GET` and `DELETE` per the transport). |
| `GET /health` | Liveness: `{ok, version, uptime_seconds}`. |
| `GET /admin/version` | Identity: name, version, scope, Smalt path, whether it exists. |
| `GET /admin/health` | The full `index_status` payload. |
| `GET /docs` | OpenAPI/Swagger UI for the HTTP routes. |

Responses are gzipped when the client accepts it. `/health` and `/admin/version` answer even
before a Smalt is bootstrapped, which is what makes the server safe to put under a process
supervisor that probes it.

### Backup and restore

There is no backup endpoint, deliberately. One shipped briefly as `GET /admin/backup` and was
removed in v0.12.0. The reasoning is that Restic's content-defined chunk-level deduplication
must see raw file content to work; a server-side tar.gz reduces every snapshot to one opaque
blob and defeats it.

The supported pattern is Restic pointed at the directory:

```sh
restic backup "$SMALT_DIR" --exclude "index/lance"
```

Excluding `index/lance/` is the recommendation, not an optimization detail: the index is
rebuildable, and including it roughly doubles snapshot size with bytes the indexer can
regenerate. This can run against a live server, since the mutex is held only briefly during
a write's commit phase; for a strict point-in-time snapshot, stop the server first.

To restore: stop the server, `restic restore latest --target /staging`, move the tree into
place, start the server against it, and rebuild the index with `reindex_all` (asynchronous;
poll the returned `task_id` with `task_status`). `bootstrap` also rebuilds and is idempotent,
but `reindex_all` is the explicit instrument for this case.

For a Smalt on a host where Restic cannot reach the filesystem, mount `SMALT_DIR` over SSHFS
and back up the mount, which preserves per-file deduplication at the cost of one hop.

### Concurrency

Three mechanisms hold concurrency together.

The **single-writer mutex** serializes the commit phase of writes and wraps the whole indexer
run, so two indexer passes fully serialize. It records contention metrics (`acquire_count`,
total and mean wait) that surface through `index_status`.

**Reads may escape the mutex safely**, and the reason is worth internalizing: LanceDB uses
snapshot-versioned reads, so a reader sees the snapshot it opened and a concurrent writer's
commit becomes visible only afterwards. The worst case is redundant work, such as skipping a
page that changed a moment ago; it is never corruption.

**Blocking work runs on a thread pool.** Synchronous handlers are wrapped and dispatched
through `asyncio.to_thread`, so a slow embedding call does not stall the event loop and
concurrent requests do not serialize behind one another.

## Failure modes and edge cases

The server is built to degrade in specific, observable ways rather than fail wholesale.

Before `bootstrap`, the server starts and stays healthy against a missing Smalt directory.
`/health` and `/admin/version` answer; tools return `smalt_not_initialized` rather than
crashing.

A failed FTS or ANN rebuild does not fail the indexer run. Page rows remain intact and
correct, and the failure is reported through `index_status` and `/admin/health` rather than
being swallowed. Search degrades to vector and alias retrieval when full-text search is
unavailable.

Below 256 embeddings the ANN index is skipped and a brute-force scan is used. This is normal
and exact, not degraded correctness.

A single malformed page, bad YAML or a path-escaping symlink, is recorded as a per-file
failure and skipped; it does not abort the run.

`remove_page` deletes index rows before the file, so a crash mid-cascade leaves a
not-yet-indexed file rather than a phantom row pointing at nothing. The former is repaired by
the next indexer pass; the latter would not be.

LanceDB's `delete(filter)` takes no bound parameters, so `sql_str()` escapes quotes before
interpolation. Without it, an id such as `x' OR '1'='1` would empty a table. Any new filter
construction must use it.

The indexer zips pages against returned vectors with `strict=True`, so an embedder returning
the wrong count fails loudly rather than silently misaligning pages and vectors.

Backing up a live server may miss a page that changed during the run; stop the server if the
snapshot must be exact.

## Design invariants

These are the commitments the code keeps, and the things to preserve when changing it.

The server is a thin, deterministic wrapper. Cognition belongs to CoGrind. If a change
requires judgment, it belongs in an agent, not here.

Markdown is canonical; the index is disposable. Any state that cannot be rebuilt from
`pages/` is a bug.

There is one writer per Smalt, and every write commits under the mutex.

Writes are atomic: a temporary file, then `os.replace`. A reader never sees a half-written
page.

The schema is forward-compatible. Models allow unknown fields, so an older server can read a
corpus written by a newer one without discarding data.

## Document currency

This document describes the code on `develop`, verified against the source rather than the
prose. Where it disagrees with `README.md`, the README lags, in three known respects:

- The README's status section reports 17 tools across three tiers (8 + 5 + 4). The registry
  and `tests/test_server.py` both hold 27 (12 + 11 + 4), and have since v0.6.0.
- The README's restore instructions call `reindex_all` "a planned future tool." It ships, in
  the `read_write` tier, and is the right instrument for a post-restore rebuild.
- The README's endpoint list omits `GET /admin/health`.

Note also that `CLAUDE.md` and `docs/CONTRIBUTING.md` instruct the reader to begin with
`docs/northstar.md`. No such file exists in this repo; the governing intent lives in
`cobalt-grinding/docs/northstar.md`, which states CoGrind's axioms and the substrate split
this server implements.

## Further reading

- `README.md` — install, run modes, and the operational quickstart.
- `CHANGELOG.md` — the release history, including the v0.5.0 substrate split.
- `cobalt-grinding/docs/northstar.md` — the governing intent: why memory is markdown, why
  memory is earned rather than stored, and how the Smalt evolves by hypothesis and test.
- `cobalt-grinding/docs/plan.md` — the full system design; the LanceDB table list here is
  kept in step with it.
- `ebony-enriching` — the sibling substrate: proposals, experiments, and gaps.
