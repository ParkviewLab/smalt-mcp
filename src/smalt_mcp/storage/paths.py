# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Canonical path conventions inside a Smalt directory."""

from __future__ import annotations

from pathlib import Path


def pages_dir(smalt_root: Path) -> Path:
    return smalt_root / "pages"


def entities_dir(smalt_root: Path) -> Path:
    return pages_dir(smalt_root) / "entities"


def concepts_dir(smalt_root: Path) -> Path:
    return pages_dir(smalt_root) / "concepts"


def source_pages_dir(smalt_root: Path) -> Path:
    return pages_dir(smalt_root) / "sources"


def syntheses_dir(smalt_root: Path) -> Path:
    return pages_dir(smalt_root) / "syntheses"


def structures_dir(smalt_root: Path) -> Path:
    return smalt_root / "structures"


def schema_dir(smalt_root: Path) -> Path:
    return smalt_root / "schema"


def schema_md_path(smalt_root: Path) -> Path:
    return schema_dir(smalt_root) / "SCHEMA.md"


def policy_md_path(smalt_root: Path) -> Path:
    return schema_dir(smalt_root) / "POLICY.md"


def index_dir(smalt_root: Path) -> Path:
    return smalt_root / "index"


def lance_dir(smalt_root: Path) -> Path:
    return index_dir(smalt_root) / "lance"


def tasks_dir(smalt_root: Path) -> Path:
    """Reserved for future Smalt-internal task state. Proposals / experiments /
    gap signals live in the `ebony-enriching` substrate (a separate MCP
    server with its own EBONY_ENRICHING_DIR), not here."""
    return smalt_root / "tasks"


def per_smalt_config_path(smalt_root: Path) -> Path:
    return smalt_root / "config.toml"


ALL_DIRS: tuple[str, ...] = (
    "pages",
    "pages/entities",
    "pages/concepts",
    "pages/sources",
    "pages/syntheses",
    "structures",
    "schema",
    "index",
    "index/lance",
    "tasks",
)
"""Directories created when bootstrapping an empty Smalt, relative to its root."""
