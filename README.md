# smalt-mcp

MCP server wrapping the **Smalt**'s storage surface (read / write / link / claim / search) for ParkviewLab's [CoGrind](https://github.com/ParkviewLab/cobalt-grinding) project. Thinnest viable wrapper around markdown + LanceDB; no agentic logic. Single-writer to a given Smalt.

To `cobalt-grinding` what [`deco-assaying`](https://github.com/ParkviewLab/deco-assaying) is to tree-sitter: a clean MCP-shaped wrapper around a deterministic capability.

## Status

**Not yet implemented.** This is **Track A of M2.7** in the CoGrind plan — see [`cobalt-grinding/docs/plan.md`](https://github.com/ParkviewLab/cobalt-grinding/blob/main/docs/plan.md) for the full design and tool surface.

## Releasing

Tag-driven via the release workflow on push of a `v*` tag. Use the [`ParkviewLab/dev-tools`](https://github.com/ParkviewLab/dev-tools) helpers — they enforce the SSOT-tag-CI loop (`pyproject.toml` is the only place the version lives; CI verifies the pushed tag matches before publishing).

```sh
git bump patch              # 0.1.5 → 0.1.6, committed
git release                 # annotated tag v0.1.6 from pyproject.toml
git push --follow-tags      # CI fires
```

Don't have the helpers? Install once: `git clone https://github.com/ParkviewLab/dev-tools.git ~/dev-tools && cd ~/dev-tools && ./install.sh`.
