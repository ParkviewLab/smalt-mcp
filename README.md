# smalt-mcp

MCP server wrapping the **Smalt**'s storage surface (read / write / link / claim / search) for ParkviewLab's [CoGrind](https://github.com/ParkviewLab/cobalt-grinding) project. Thinnest viable wrapper around markdown + LanceDB; no agentic logic. Single-writer to a given Smalt.

To `cobalt-grinding` what [`deco-assaying`](https://github.com/ParkviewLab/deco-assaying) is to tree-sitter: a clean MCP-shaped wrapper around a deterministic capability.

## Status

**v0 foundation.** Server runs; one MCP tool (`status`) wired up; storage layer ported from `cobgrind/storage/` and ready for the rest of the tool surface. Track A of CoGrind's M2.7 plan — see [`cobalt-grinding/docs/plan.md`](https://github.com/ParkviewLab/cobalt-grinding/blob/main/docs/plan.md) for the full design.

Coming in the next iterations: read-only tools (`list_pages`, `read_page`, `search`, `traverse`, `list_domains`, `list_proposals`), read-write tools (`write_page`, `write_pages`, `add_link`, `add_claim`, `write_proposal`, `bootstrap`), auto-indexer-trigger on writes, end-to-end integration with CoGrind.

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

## MCP tools (v0)

**Read-only:**

- `status` — Smalt path, existence, LanceDB tables present, page count, single-writer mutex state, embedding provider. Always safe to call.

**Read-write:** *(coming next iteration)*

The full target surface is in [`cobalt-grinding/docs/plan.md`](https://github.com/ParkviewLab/cobalt-grinding/blob/main/docs/plan.md) under "Track A — `ParkviewLab/smalt-mcp` v0".

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `35833` | HTTP listen port. |
| `HOST` | `0.0.0.0` | HTTP bind address. |
| `SMALT_DIR` | `~/Documents/Smalt` | Path to the Smalt this server wraps. Auto-created on bootstrap (planned). |
| `SMALT_SCOPE` | `read_write` | `read_only` or `read_write`. Read-only deployments only expose tools that don't mutate the corpus. |
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
