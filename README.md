<!--
SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

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

## Operations: backup and restore

The Smalt is a directory of markdown files (plus a rebuildable LanceDB index). **Use [Restic](https://restic.net/) directly against `SMALT_DIR`** for backup — there's no dedicated backup endpoint on the server, and it's intentional: Restic's content-defined chunk-level deduplication needs to see raw file content. Pointing it at the live directory gives real per-file dedup, real incremental snapshots, and a restore-as-directory-tree workflow that's strictly better than any server-side archive-export endpoint would deliver.

### Backup

```sh
restic backup "$SMALT_DIR" --exclude "index/lance"
```

Excluding `index/lance/` is the recommended default — the LanceDB store is rebuildable from the markdown in `pages/`. Including it would roughly double snapshot size with bytes that the indexer can regenerate post-restore.

You can run this against a live smalt-mcp (the server only holds the corpus mutex briefly, during the commit phase of a write). For a strictly point-in-time snapshot, stop the server first.

### Restore

```sh
# 1. Stop smalt-mcp (otherwise it could race the restore).
# 2. Restore from the latest snapshot to a staging dir.
restic restore latest --target /staging

# 3. Move the restored Smalt into place.
mv /staging/<path-restic-recorded>/Smalt "$SMALT_DIR"

# 4. Start smalt-mcp pointing at the restored SMALT_DIR.
SMALT_DIR="$SMALT_DIR" uv run python -m smalt_mcp   # or your usual run mode
```

Then trigger an index rebuild via the MCP `bootstrap` tool. `bootstrap` is idempotent — it'll detect the restored markdown and rebuild the LanceDB index from it (since we excluded `index/lance/` from the backup). A planned future tool (`reindex_all`) will be the cleaner explicit version of this for the restore use case; until it ships, `bootstrap` is the right call.

### Remote Smalts (running on a host where Restic can't reach the filesystem)

Mount `SMALT_DIR` locally via SSHFS (or equivalent), then `restic backup` against the mount. Same per-file dedup as the local case, with one extra hop. If even SSHFS isn't possible (very restricted deployment), the prior approach of building a tar.gz server-side and piping to `restic backup --stdin` is technically possible but defeats Restic's dedup — every snapshot becomes one opaque binary blob. Not recommended.

### Why no `/admin/backup` endpoint

Earlier iterations of smalt-mcp briefly shipped a `GET /admin/backup` streaming tar.gz endpoint. It was removed in v0.12.0 after we realized the Restic-native pattern is strictly better for the common case. The endpoint design (streaming tar.gz via stdlib `tarfile`, best-effort consistency, scope-filtered downloads) was sound; the question was whether to ship a half-good answer (opaque blob, zero dedup) or the right answer (Restic against the filesystem). We chose the latter.

## Releasing

Tag-driven via the release workflow on push of a `v*` tag. Use the [`ParkviewLab/dev-tools`](https://github.com/ParkviewLab/dev-tools) helpers — they enforce the SSOT-tag-CI loop (`pyproject.toml` is the only place the version lives; CI verifies the pushed tag matches before publishing).

```sh
git bump patch              # 0.1.5 → 0.1.6, committed
git release                 # annotated tag v0.1.6 from pyproject.toml
git push --follow-tags      # CI fires
```

Don't have the helpers? Install once: `git clone https://github.com/ParkviewLab/dev-tools.git ~/dev-tools && cd ~/dev-tools && ./install.sh`.

### Commit message convention

After the publish jobs, a **changelog** job generates the new `CHANGELOG.md` section — an LLM-written "Highlights" paragraph plus a [`git-cliff`](https://git-cliff.org/) categorized commit list — commits it back to `main`, and creates the GitHub Release with the same content as its body. Categorization uses [Conventional Commits](https://www.conventionalcommits.org/) prefixes (see [`cliff.toml`](cliff.toml)):

| Prefix | Section | Notes |
|---|---|---|
| `feat:` | Features | user-visible |
| `fix:` | Bug fixes | user-visible |
| `perf:` | Performance | user-visible |
| `refactor:` | Refactor | |
| `docs:` | Docs | |
| `test:` | Tests | |
| `chore:` / `ci:` / `build:` / `style:` | _(dropped)_ | not surfaced in CHANGELOG |

Squash-merge PRs use the PR title as the commit subject — so the **PR title** is what needs the prefix. Commits without a recognised prefix are silently dropped from the CHANGELOG (still in git history). The "Highlights" paragraph requires the `ANTHROPIC_API_KEY` org-level secret; if the LLM call fails, a placeholder lands and the release still ships.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>), or
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option. In SPDX terms: `MIT OR Apache-2.0`.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you shall be dual-licensed as above, without any
additional terms or conditions. See [LICENSING.md](LICENSING.md).

---
<sub>© 2026 Gary Frattarola · Licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) · part of [ParkviewLab](https://github.com/ParkviewLab)</sub>
