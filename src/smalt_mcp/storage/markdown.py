# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Read and parse markdown pages with YAML frontmatter.

Each page in `smalt/pages/<type>/<id>.md` has a YAML frontmatter block delimited
by `---` lines, followed by the page body. The frontmatter validates against
the Pydantic models in `smalt_mcp.schema`; the body is plain prose fed to FTS
and (truncated) to the embedder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml
from pydantic import TypeAdapter

from smalt_mcp.schema import Page

PAGE_ADAPTER: TypeAdapter[Page] = TypeAdapter(Page)


@dataclass(frozen=True)
class ParsedPage:
    """A markdown page, parsed and validated.

    Attributes
    ----------
    path: absolute path to the .md file
    rel_path: path relative to `smalt_root` (used as the page's stable handle)
    frontmatter: parsed-and-validated `Page` (one of the discriminated union members)
    body: page body (everything after the frontmatter block)
    content_hash: sha256 of the file's bytes — used for incremental indexing
    raw_frontmatter: the dict that came out of the YAML parse, before model validation;
        useful for storing as JSON in LanceDB without re-serializing
    """

    path: Path
    rel_path: str
    frontmatter: Page
    body: str
    content_hash: str
    raw_frontmatter: dict[str, Any]


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_page(path: Path, *, smalt_root: Path) -> ParsedPage:
    """Read, parse, and validate one markdown page.

    Raises
    ------
    ValidationError: if the frontmatter doesn't match any of the page-type schemas.
    ValueError: if the file has no frontmatter block, or the frontmatter is
        malformed YAML.
    """
    raw_bytes = path.read_bytes()
    content_hash = hash_bytes(raw_bytes)

    try:
        post = frontmatter.loads(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as e:
        # python-frontmatter delegates YAML parsing to PyYAML; re-raise as
        # ValueError so the indexer's narrow except tuple catches it instead
        # of letting one malformed page take down a whole index run.
        raise ValueError(f"{path}: malformed YAML frontmatter — {e}") from e

    if not post.metadata:
        raise ValueError(f"{path}: no frontmatter found")

    page = PAGE_ADAPTER.validate_python(post.metadata)

    try:
        rel_path = str(path.resolve().relative_to(smalt_root.resolve()))
    except ValueError as e:
        # path.resolve() follows symlinks; if the target is outside smalt_root
        # `relative_to` raises ValueError with an opaque "not in the subpath"
        # message. Surface a clearer one — the Indexer's outer except tuple
        # will catch this and record the file as failed.
        raise ValueError(
            f"{path}: resolves to a location outside the Smalt "
            f"({path.resolve()}); symlinks outside the Smalt tree are not allowed"
        ) from e

    return ParsedPage(
        path=path,
        rel_path=rel_path,
        frontmatter=page,
        body=post.content,
        content_hash=content_hash,
        raw_frontmatter=dict(post.metadata),
    )


def iter_page_files(pages_root: Path) -> list[Path]:
    """Return all .md files under `pages_root`, sorted for deterministic ordering."""
    if not pages_root.exists():
        return []
    return sorted(p for p in pages_root.rglob("*.md") if p.is_file())
