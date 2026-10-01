<!--
SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# smalt-mcp

MCP server wrapping the **Smalt**'s storage surface (read / write / link / claim / search) for ParkviewLab's [CoGrind](https://github.com/ParkviewLab/cobalt-grinding) project. Thinnest viable wrapper around markdown + LanceDB; no agentic logic. Single-writer to a given Smalt.

To `cobalt-grinding` what [`deco-assaying`](https://github.com/ParkviewLab/deco-assaying) is to tree-sitter: a clean MCP-shaped wrapper around a deterministic capability.

## Status

**Storage substrate complete.** The full storage-substrate surface is wired up: 27 tools across three permission tiers, with auto-indexer-trigger on writes and hybrid (FTS + vector + alias, RRF-fused) search. See [`cobalt-grinding/docs/architecture.md`](https://github.com/ParkviewLab/cobalt-grinding/blob/main/docs/architecture.md) for how CoGrind uses it.

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

Bootstrap, every write and `reindex_all` need two settings: `SMALT_INTERNAL_TOKEN` set to a non-empty value, and `SMALT_SCOPE` at `read_write` or above (the default is `read_write`). Without the token the scope is capped at `read_only` whatever `SMALT_SCOPE` says, and an empty token counts as unset. The examples below set the token to a placeholder; use your own value.

### `uvx` (one-off)

```bash
SMALT_DIR=~/Documents/Smalt SMALT_INTERNAL_TOKEN=CHANGE-ME uvx smalt-mcp
```

### `uv tool install` (pinned daemon)

```bash
uv tool install smalt-mcp
SMALT_DIR=~/Documents/Smalt SMALT_INTERNAL_TOKEN=CHANGE-ME smalt-mcp
```

### From source

```bash
git clone https://github.com/ParkviewLab/smalt-mcp.git
cd smalt-mcp
uv sync
SMALT_DIR=~/Documents/Smalt SMALT_INTERNAL_TOKEN=CHANGE-ME uv run python -m smalt_mcp
```

### Docker

```bash
docker pull ghcr.io/parkviewlab/smalt-mcp:latest
# To bootstrap or write, both settings are needed: SMALT_SCOPE at read_write
# or above, and a non-empty SMALT_INTERNAL_TOKEN. Replace the -e line below with:
#   -e SMALT_SCOPE=read_write -e SMALT_INTERNAL_TOKEN=CHANGE-ME
docker run --rm \
  -p 35833:35833 \
  -e SMALT_SCOPE=read_only \
  -v smalt-data:/data \
  ghcr.io/parkviewlab/smalt-mcp:latest
```

Or use [`docker-compose.yml`](docker-compose.yml), which carries a commented `SMALT_INTERNAL_TOKEN` line and the same note.

## Endpoints

- `POST /sse` — MCP Streamable HTTP transport. Tools. `/sse` also answers `GET` and `DELETE`, as the transport requires.
- `GET /health` — liveness probe (`{ok, version, uptime_seconds}`).
- `GET /admin/version` — server identity + scope + configured Smalt path.
- `GET /admin/health` — the full `index_status` payload (table row counts, last index result, FTS and ANN status, mutex contention).
- `GET /docs` — OpenAPI / Swagger UI for the HTTP routes.

HTTP responses are gzipped when the client sends `Accept-Encoding: gzip`.

## MCP tools

Three permission tiers controlled by `SMALT_SCOPE`. A caller at tier N sees and may call any tool whose required scope is ≤ N. While `SMALT_INTERNAL_TOKEN` is unset, the scope is capped at `read_only` whatever `SMALT_SCOPE` says (an unconfigured token means read-only), and, when the cap lowers the requested scope, the server logs a warning naming it; set the token to serve `read_write` or `remove_destructive`.

**`read_only` (12 tools):**

- `status` — Smalt path, existence, LanceDB tables, page count, single-writer mutex state, embedding provider.
- `index_status` — deep health: per-table row counts, last index result, per-field FTS status, ANN status, embedding config, mutex contention. Same payload as `GET /admin/health`.
- `list_pages` — indexed pages, filtered by `type` / `prefix` plus property filters; reports `truncated`.
- `read_page` — full page (frontmatter + body); resolves by exact id, then exact alias, then fuzzy alias.
- `find_by_alias` — every page whose `aliases` list contains the given alias, exact then fuzzy.
- `incoming_links` — "what links to this page" (the inverse of `traverse`).
- `traverse` — breadth-first walk of outgoing edges from a page; optional label filter; `hops` defaults to 1 and is capped at 5.
- `search` — hybrid FTS + vector + alias, RRF-fused; every hit carries `id`, `aliases`, `title`, `type`, `snippet`, `score`.
- `list_domains` — ConceptPages flagged `is_domain: true`.
- `source_similarity` — pages most similar to a source's stored embedding, by cosine similarity.
- `task_status` — state of one async task.
- `task_list` — async tasks, filtered by state or kind.

**`read_write` (+11 tools):**

- `bootstrap` — initialize the canonical layout + LanceDB tables; idempotent.
- `write_page` — `create` (always-mangle: caller-id becomes slug-prefix + 22-char UUID4 suffix; original id preserved in aliases) or `update` (requires existing canonical id). Runs the incremental indexer.
- `write_pages` — batch of writes; validate-all-then-act; single indexer pass at the end.
- `add_link` — append an outgoing link to a page's `links_out`; duplicate detection.
- `add_claim` — append a `Claim` to a page's `claims`; duplicate-id detection.
- `add_links` — batch form of `add_link`: one read, one write, one indexer pass.
- `add_claims` — batch form of `add_claim`.
- `write_batch` — a mixed-operation atomic transaction over page writes, links and claims; validate, existence-check, commit. Cross-operation references within one batch are not supported.
- `reindex_page` — force one page to be re-indexed from disk.
- `reindex_all` — wipe and rebuild the whole index. Asynchronous: returns a `task_id` to poll with `task_status`.
- `task_cancel` — cooperatively cancel an async task.

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
| `SMALT_DIR` | `~/Documents/Smalt` | Path to the Smalt this server wraps. Call the `bootstrap` MCP tool once to initialize (it needs `SMALT_INTERNAL_TOKEN` set and `SMALT_SCOPE` at `read_write` or above; see Run). |
| `SMALT_SCOPE` | `read_write` | `read_only`, `read_write`, or `remove_destructive`. Tiered: caller at tier N sees every tool whose required scope is ≤ N. Capped at `read_only` while `SMALT_INTERNAL_TOKEN` is unset. |
| `EMBEDDING_PROVIDER` | `fastembed` | Embedding backend. `fastembed` is the only one wired up; `voyage` / `openai` are placeholders. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Model name passed to the provider. |
| `EMBEDDING_DIM` | `384` | Must match the model. |
| `SMALT_THREAD_POOL_WORKERS` | `32` | Bounds concurrent handler execution on the loop's thread pool. |
| `SMALT_FUZZY_ALIAS_THRESHOLD` | `0.6` | Trigram-Jaccard threshold for fuzzy alias resolution. |
| `SMALT_INTERNAL_TOKEN` | *(unset)* | Unset or empty: the server is read-only whatever `SMALT_SCOPE` says. Set: `SMALT_SCOPE` applies. Not checked on incoming requests; it only lifts the cap. |

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
SMALT_DIR="$SMALT_DIR" SMALT_INTERNAL_TOKEN=CHANGE-ME uv run python -m smalt_mcp   # or your usual run mode
```

Then trigger an index rebuild via the MCP `reindex_all` tool (the `read_write` tier, so the server needs `SMALT_INTERNAL_TOKEN` set and `SMALT_SCOPE` at `read_write` or above). It is asynchronous: it returns a `task_id`, which `task_status` polls. It wipes the LanceDB tables and rebuilds them from the restored markdown (since we excluded `index/lance/` from the backup). `bootstrap` is idempotent and also rebuilds the index, but `reindex_all` is the explicit instrument for the restore use case.

### Remote Smalts (running on a host where Restic can't reach the filesystem)

Mount `SMALT_DIR` locally via SSHFS (or equivalent), then `restic backup` against the mount. Same per-file dedup as the local case, with one extra hop. If even SSHFS isn't possible (very restricted deployment), the prior approach of building a tar.gz server-side and piping to `restic backup --stdin` is technically possible but defeats Restic's dedup — every snapshot becomes one opaque binary blob. Not recommended.

### Why no `/admin/backup` endpoint

A `GET /admin/backup` streaming tar.gz endpoint was built during development and reverted before any release, after we realized the Restic-native pattern is strictly better for the common case. The endpoint design (streaming tar.gz via stdlib `tarfile`, best-effort consistency, scope-filtered downloads) was sound; the question was whether to ship a half-good answer (opaque blob, zero dedup) or the right answer (Restic against the filesystem). We chose the latter.

## Releasing

Tag-driven via the release workflow on push of a `v*` tag. Use the [`ParkviewLab/dev-tools`](https://github.com/ParkviewLab/dev-tools) helpers — they enforce the SSOT-tag-CI loop (`pyproject.toml` is the only place the version lives; CI verifies the pushed tag matches before publishing).

Run the sequence in the `smalt-mcp-main` worktree, under the release's authorisation (see the handbook's [`releases.md`](https://github.com/ParkviewLab/handbook/blob/main/docs/releases.md)):

```sh
git pull --ff-only                          # sync main
git -C ../smalt-mcp-develop pull --ff-only  # sync develop too
git merge --no-ff develop                   # promote develop to main
git bump <kind>                             # patch, minor, major, release or X.Y.Z; commits "release v<new>"
git release                                 # annotated tag v<new> from pyproject.toml
git push --follow-tags                      # CI fires
git back-merge                              # the last step: bring the release into develop
```

Don't have the helpers? Install once: `git clone https://github.com/ParkviewLab/dev-tools.git ~/dev-tools && cd ~/dev-tools && ./install.sh`.

### Commit message convention

After the publish jobs, a **changelog** job generates the new `CHANGELOG.md` section — an LLM-written "Highlights" paragraph plus a categorized list written by dev-tools' `generate-changelog` — commits it back to `main`, and creates the GitHub Release with the same content as its body. Categorization uses [Conventional Commits](https://www.conventionalcommits.org/) prefixes (the full list is in the ParkviewLab handbook's `commits-and-changelogs.md`):

| Title | Group in the notes | Notes |
|---|---|---|
| any type with `!` after it (`feat!:`), or a breaking-change footer | Breaking changes | listed there once, whatever its type |
| `feat:` | Features | user-visible |
| `fix:` | Bug fixes | user-visible |
| `perf:` | Performance | user-visible |
| `refactor:` | Refactor | |
| `docs:` | Docs | |
| `test:` | Tests | |
| `revert:` | Reverts | GitHub's Revert button titles a PR `Revert "…"`, which has no type |
| `build:` / `chore:` / `ci:` / `style:` | Maintenance | |
| any other title | Other changes | the whole title |
| a commit with no pull request | Direct commits | its subject and short hash |

A title without a recognised type is not dropped: it is listed whole under Other changes. So prefix your PR titles, and correct a title before the merge, since retitling afterwards does not change the commit. The groups appear in the order above, and an empty group is left out.

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
