# smalt-mcp

MCP server wrapping the **Smalt**'s storage surface (read / write / link / claim / search) for ParkviewLab's [CoGrind](https://github.com/ParkviewLab/cobalt-grinding) project. Thinnest viable wrapper around markdown + LanceDB; no agentic logic. Single-writer to a given Smalt.

To `cobalt-grinding` what [`deco-assaying`](https://github.com/ParkviewLab/deco-assaying) is to tree-sitter: a clean MCP-shaped wrapper around a deterministic capability.

## Status

**Storage substrate complete.** The full storage-substrate surface is wired up: 17 tools across three permission tiers, with auto-indexer-trigger on writes and hybrid (FTS + vector + alias, RRF-fused) search. Track A of CoGrind's M2.7 plan — see [`cobalt-grinding/docs/plan.md`](https://github.com/ParkviewLab/cobalt-grinding/blob/main/docs/plan.md) for the full design.

The scientific-method surface (proposals / experiments / gaps) is **not** part of smalt-mcp — it lives in a separate MCP server, [`ebony-enriching`](https://github.com/ParkviewLab/ebony-enriching), the lab-notebook substrate. Cobalt-grinding's cognitive systems read from both substrates and orchestrate cross-substrate writes.

## Run

Same five-mode pattern as deco-assaying. Pick whichever fits.

| Mode | When to use |
|---|---|
| 1. `uvx` (one-off) | Try it once, no install. |
| 2. `uv tool install` (pinned daemon) | Run it occasionally, want it on `$PATH`. |
| 3. macOS LaunchAgent | Persistent daemon on a Mac. |
| 4. Linux systemd user unit | Persistent daemon on Linux. |
| 5. Docker / docker compose | Container deployment. |

In every mode the server listens on `PORT` (default `35833`). Sanity-check:

```bash
curl http://127.0.0.1:35833/health
```

### From source (current; until first release)

```bash
git clone https://github.com/ParkviewLab/smalt-mcp.git
cd smalt-mcp
uv sync
SMALT_DIR=~/Documents/Smalt uv run python -m smalt_mcp
```

### Docker (after first release)

```bash
docker pull ghcr.io/parkviewlab/smalt-mcp:latest
docker run --rm \
  -p 35833:35833 \
  -e SMALT_SCOPE=read_only \
  -v smalt-data:/data \
  ghcr.io/parkviewlab/smalt-mcp:latest
```

Or use [`docker-compose.yml`](docker-compose.yml).

## Endpoints

- `POST /sse` — MCP Streamable HTTP transport. Tools.
- `GET /health` — liveness probe (`{ok, version, uptime_seconds}`).
- `GET /admin/version` — server identity + scope + configured Smalt path.
- `GET /docs` — OpenAPI / Swagger UI for the HTTP routes.

HTTP responses are gzipped when the client sends `Accept-Encoding: gzip`.

## MCP tools

Three permission tiers controlled by `SMALT_SCOPE`. A caller at tier N sees and may call any tool whose required scope is ≤ N.

**`read_only` (8 tools):**

- `status` — Smalt path, existence, LanceDB tables, page count, single-writer mutex state, embedding provider.
- `list_pages` — indexed pages, filtered by `type` / `prefix`.
- `read_page` — full page (frontmatter + body); falls back to alias lookup on miss.
- `find_by_alias` — every page whose `aliases` list contains the given alias.
- `incoming_links` — "what links to this page" (the inverse of `traverse`).
- `traverse` — outgoing edges from a page; optional label filter.
- `search` — hybrid FTS + vector + alias, RRF-fused; every hit carries `id`, `aliases`, `title`, `type`, `snippet`, `score`.
- `list_domains` — ConceptPages flagged `is_domain: true`.

**`read_write` (+5 tools):**

- `bootstrap` — initialize the canonical layout + LanceDB tables; idempotent.
- `write_page` — `create` (always-mangle: caller-id becomes slug-prefix + 22-char UUID4 suffix; original id preserved in aliases) or `update` (requires existing canonical id). Runs the incremental indexer.
- `write_pages` — batch of writes; validate-all-then-act; single indexer pass at the end.
- `add_link` — append an outgoing link to a page's `links_out`; duplicate detection.
- `add_claim` — append a `Claim` to a page's `claims`; duplicate-id detection.

**`remove_destructive` (+4 tools):**

- `remove_page` — cascading delete (file + pages row + embeddings row + outgoing + incoming links + claims).
- `update_claim` — replace one claim by id; `new_claim.id` must equal `claim_id`.
- `remove_claim` — remove one claim by id.
- `remove_link` — remove edges by `(from_id, to_id, label?)`; omit `label` to drop every edge between the pair.

For the proposal / experiment / gap surface (writing hypotheses, recording experiment runs, queueing knowledge gaps), use [`ebony-enriching`](https://github.com/ParkviewLab/ebony-enriching) — the lab-notebook substrate. Both servers are independent: cobalt-grinding's cognitive systems orchestrate any cross-substrate flow.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `35833` | HTTP listen port. |
| `HOST` | `0.0.0.0` | HTTP bind address. |
| `SMALT_DIR` | `~/Documents/Smalt` | Path to the Smalt this server wraps. Call the `bootstrap` MCP tool once to initialize. |
| `SMALT_SCOPE` | `read_write` | `read_only`, `read_write`, or `remove_destructive`. Tiered: caller at tier N sees every tool whose required scope is ≤ N. |
| `EMBEDDING_PROVIDER` | `fastembed` | Embedding backend. `fastembed` is the only one wired up; `voyage` / `openai` are placeholders. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Model name passed to the provider. |
| `EMBEDDING_DIM` | `384` | Must match the model. |
| `SMALT_INTERNAL_TOKEN` | *(unset)* | Reserved for future per-client scope routing; not yet enforced. |

## Releasing

Tag-driven via the release workflow on push of a `v*` tag. Use the [`ParkviewLab/dev-tools`](https://github.com/ParkviewLab/dev-tools) helpers — they enforce the SSOT-tag-CI loop (`pyproject.toml` is the only place the version lives; CI verifies the pushed tag matches before publishing).

```sh
git bump patch              # 0.1.5 → 0.1.6, committed
git release                 # annotated tag v0.1.6 from pyproject.toml
git push --follow-tags      # CI fires
```

Don't have the helpers? Install once: `git clone https://github.com/ParkviewLab/dev-tools.git ~/dev-tools && cd ~/dev-tools && ./install.sh`.

## License

MIT. See `LICENSE`.
