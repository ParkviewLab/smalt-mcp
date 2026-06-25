<!--
SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Changelog

All notable changes to this project are recorded here. Each release entry
has two parts:

- **Highlights** — a 2-3 sentence "what's new" paragraph generated at
  release time by an Anthropic-API call (see
  `scripts/generate_changelog.py`).
- **Categorized changes** — a list of merged commits since the previous
  tag, grouped by [Conventional Commit](https://www.conventionalcommits.org/)
  prefix, produced by [git-cliff](https://git-cliff.org/) using
  `cliff.toml`.

The release workflow on every tag push regenerates both, commits the new
section here, and uses the same content as the GitHub Release body.

<!--
  Keep-a-Changelog ordering: [Unreleased] at the top, then newest
  released version, then older versions. generate_changelog.py inserts
  new "## [vX.Y.Z] - YYYY-MM-DD" sections directly below [Unreleased].
  Don't remove the marker.
-->

## [Unreleased]

## [v1.3.3] - 2026-06-25

### Highlights

This is a maintenance release that bumps lagging GitHub Actions pins to their Node 24 floors, with no user-facing behavior changes.

### Docs

- V1.3.2 [skip ci] (3dee2ea)

## [v1.3.2] - 2026-06-15

### Highlights

This is a security maintenance release that floors the starlette dependency to >=1.0.1 to address GHSA-86qp-5c8j-p5mr, a Host-header validation issue in the previously unconstrained 1.0.0 resolution.

### Bug fixes

- Floor starlette>=1.0.1 (GHSA-86qp-5c8j-p5mr) (#43) (770d64c)

### Docs

- V1.3.1 [skip ci] (1bbe671)

## [v1.3.1] - 2026-06-15

### Highlights

This is a maintenance release that aligns smalt-mcp with ParkviewLab's handbook conventions, with no behavioural changes. The license moves from MIT to a dual `MIT OR Apache-2.0` arrangement, with dual LICENSE files, per-file SPDX headers, a REUSE.toml, and updated README and LICENSING.md. Internally, CI workflows for reuse, version-guard, test, license-check, and a dormant dev-release were added, ruff and ty replaced the prior lint/type setup, and the test suite (217 tests) remains green.

### Docs

- V1.3.0 [skip ci] (8941151)

## [v1.3.0] - 2026-06-14

### Highlights

This release adds an automated changelog and GitHub Release generation step to the tag-push workflow, producing a CHANGELOG.md section (LLM-written highlights paragraph plus a git-cliff categorized commit list) and publishing it as the GitHub Release body. CHANGELOG.md has been backfilled for all prior tags from v0.1.0 through v1.2.0, and the README now documents the changelog job and the Conventional Commit convention it relies on. The workflow requires an org-level ANTHROPIC_API_KEY secret, and falls back to a placeholder paragraph if it is unset.

## [v1.2.0] - 2026-05-18

### Highlights

This release adds a `--transport` flag allowing smalt-mcp to run as a stdio MCP server, so it can be launched as a subprocess directly from configurations like `claude_desktop_config.json` or `mcp.json` without running an HTTP daemon. The default transport remains http, leaving existing Docker, LaunchAgent, and systemd deployments unaffected.

## [v1.1.0] - 2026-05-17

### Highlights

Every tool handler now dispatches its blocking work via `asyncio.to_thread`, so concurrent MCP requests run with true parallelism for reads and single-writer serialization (via the corpus mutex) for writes; the event loop stays free regardless of which handler is in flight. Thread-pool sizing is configurable via the new `SMALT_THREAD_POOL_WORKERS` environment variable (default 32). This release also folds in the v1.0.1 concurrency-hardening fixes: locked double-checked init for the LanceDB connection and embedder, a lock around the observability `last_*` fields to prevent torn reads, and eager pre-warming of both resources during server lifespan startup.

## [v1.0.0] - 2026-05-16

### Highlights

This release completes Workstream C, capping it with an in-process async task scheduler so that heavy operations like `reindex_all` no longer block the request handler — callers now receive a `task_id` immediately and poll via the new `task_status`, `task_list`, and `task_cancel` tools, which is a breaking change to `reindex_all`'s response shape. Also new since v0.5.0: explicit `reindex_page`/`reindex_all` tools, a `source_similarity` vector-search tool, fuzzy alias fallback on `find_by_alias` and `read_page`, bulk `add_links`/`add_claims` and mixed-op `write_batch` transactions, auto-regenerated IndexPages (glossary, domains), the `<source-id>::<rel-path>` section-page id format, and six new frontmatter property filters on `list_pages` and `search`. Operability gains include a `/admin/health` endpoint and matching `index_status` MCP tool that surfaces previously-silent FTS/ANN failures, a promoted `aliases` LanceDB column for O(1) alias lookups, and a documented Restic-native backup pattern (the earlier `/admin/backup` endpoint was reverted before release). Tool count grew from 17 to 27 (12 read-only, 11 read-write, 4 remove-destructive).

## [v0.5.0] - 2026-05-16

### Highlights

This release narrows smalt-mcp to a pure storage substrate by removing the proposal, experiment, and gap surface, which is moving to a separate MCP server called ebony-enriching. The `write_proposal` and `list_proposals` tools are gone, leaving 17 tools (8 read-only, 5 read-write, 4 remove-destructive), and the associated schema models, bootstrap directories (`tasks/proposals/`, `tasks/gaps.md`), and path entries have been deleted. This is a breaking change for callers of the removed tools, which must migrate to the forthcoming `ebony.*` surface.

## [v0.4.0] - 2026-05-16

### Highlights

Search now matches aliases as a third retrieval source alongside FTS and vector similarity, so a page whose alias appears in the query (verbatim or as a whitespace-separated token) will surface even when its title and body share nothing with the search terms. Search hits also now include the `aliases` list per result, letting callers render results by their memorable handle without a follow-up `find_by_alias` or `read_page` call. The `/health` and `/admin/version` HTTP endpoints gained OpenAPI tags for consistent grouping in the auto-generated `/docs` and `/redoc` pages.

## [v0.3.0] - 2026-05-16

### Highlights

This release adds an incoming_links tool for auditing what points at a page, plus a destructive surface — remove_page (cascading delete of file, links, claims, and embedding), update_claim, remove_claim, and remove_link — bringing the total to 19 tools. A new three-tier permission model (read_only, read_write, remove_destructive) gated by the SMALT_SCOPE env var lets operators choose whether the destructive tools are exposed at all, with read_write remaining the default.

## [v0.2.0] - 2026-05-16

### Highlights

The write surface now enforces a portable, path-traversal-safe id shape (rejecting `..`, slashes, Windows-reserved names, and other unsafe forms) on both `write_page` and `write_proposal`. Page creation is now always-mangle: `write_page` in `create` mode assigns a canonical id of the form `<caller-id>-<22-char-base64-UUID>` and preserves the caller's original id in the page's `aliases` list, while `update` mode requires an existing canonical id; the previous `upsert` mode is gone. To make alias-based addressing usable, a new `find_by_alias` tool returns all pages carrying a given alias, and `read_page` falls back to alias lookup when an exact id miss yields a single match (returning `ambiguous_alias` with the candidate list when more than one matches).

## [v0.1.0] - 2026-05-16

### Highlights

Initial release of the smalt-mcp server, exposing Smalt's storage surface over MCP with a 13-tool v0 API: seven read-only tools (status, list_pages, read_page, traverse, search, list_domains, list_proposals) and six read-write tools (bootstrap, write_page, write_pages, write_proposal, add_link, add_claim). Writes are serialized through a single-writer corpus mutex with atomic tmp-then-rename and an incremental indexer pass, search is hybrid FTS + vector with RRF fusion, and tool scope is gated by SMALT_SCOPE / SMALT_INTERNAL_TOKEN. Ships with a Dockerfile, docker-compose, and a release workflow with tag/pyproject gating.

