"""Pydantic models for the page frontmatter schema.

These are the canonical types for everything that lives in `smalt/pages/`.
The indexer parses each page's YAML frontmatter into one of these models
and projects them into LanceDB; agents producing new pages round-trip
through these models so that frontmatter conforms by construction.

The schema is intentionally permissive — fields are mostly optional with
sane defaults — and will tighten as the Smalt's structure stabilizes.
SCHEMA.md (in each Smalt directory) is the human-facing narrative version
of these rules; this module is the machine-facing one.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- enums ----


class PageType(StrEnum):
    """Top-level kind of Smalt page. Each type has its own frontmatter shape."""

    ENTITY = "entity"
    CONCEPT = "concept"
    SOURCE = "source"
    SYNTHESIS = "synthesis"  # cross-source pages, written by Cogitate (Phase 2)


class LocationKind(StrEnum):
    """How a source's `location_uri` should be interpreted."""

    FILE = "file"
    DIR = "dir"
    GIT = "git"
    OBSIDIAN = "obsidian"
    URL = "url"  # Phase 2


class ConfidenceLevel(StrEnum):
    """Coarse confidence label. Numeric confidences live alongside as floats."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRATED = "unrated"


# ---- shared building blocks ----


class Link(BaseModel):
    """A directed, optionally-labeled edge from one page to another."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="page-id or Smalt path of the linked page")
    label: str | None = Field(default=None, description="optional edge label")
    via_source: str | None = Field(
        default=None,
        description="source-id this edge was derived from, if applicable",
    )


class Claim(BaseModel):
    """A structured assertion attached to a page."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    value_type: Literal["string", "number", "bool", "date"] | None = None
    value: str | float | bool | None = None
    unit: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_label: ConfidenceLevel = ConfidenceLevel.UNRATED
    source_ref: str | None = Field(
        default=None,
        description="source pointer, e.g. `git:<remote>@<sha>:<path>:Lx-Ly` or `file:<path>#<heading>`",
    )


# ---- common base for all page types ----


class PageBase(BaseModel):
    """Fields shared by every page type."""

    model_config = ConfigDict(extra="allow")  # allow forward-compat keys we don't model yet

    id: str = Field(description="stable Smalt-internal id; usually a slug")
    type: PageType
    title: str
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    created_at: datetime | None = None
    updated_at: datetime | None = None

    links_out: list[Link] = Field(default_factory=list)

    # Reserved for the future veracity / quality system. Populated as
    # `null` / `unrated` until then; defining now to avoid a schema migration later.
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    veracity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evaluated_at: datetime | None = None
    evaluation_notes: str | None = None


# ---- per-type page models ----


class EntityPage(PageBase):
    type: Literal[PageType.ENTITY] = PageType.ENTITY  # type: ignore[assignment]
    entity_kind: str | None = Field(
        default=None,
        description="optional sub-kind: person, org, product, repo, package, place, ...",
    )
    claims: list[Claim] = Field(default_factory=list)


class ConceptPage(PageBase):
    type: Literal[PageType.CONCEPT] = PageType.CONCEPT  # type: ignore[assignment]
    parents: list[str] = Field(
        default_factory=list,
        description="ids of broader concept pages (taxonomy edges, populated by Cogitate)",
    )
    claims: list[Claim] = Field(default_factory=list)


class SourcePage(PageBase):
    """A page that represents one ingested source.

    The bytes of the source are NOT stored anywhere in the Smalt. This page
    is the Smalt's notes *about* the source, plus a stable pointer to where
    it lives.
    """

    type: Literal[PageType.SOURCE] = PageType.SOURCE  # type: ignore[assignment]

    # Where the source lives.
    location_uri: str = Field(
        description="stable pointer: file:<path> | dir:<path> | git:<remote-url> | obsidian:<path> | url:<...>"
    )
    location_kind: LocationKind

    # Drift detection.
    source_content_hash: str | None = Field(
        default=None,
        description="hash (sha256) of the source content at fetch time, or null if not applicable (e.g. directory)",
    )
    fetched_at: datetime | None = None
    last_verified_at: datetime | None = None

    # Original organization of the source, preserved as provenance.
    structure_inline: dict[str, object] | None = Field(
        default=None,
        description="small structures (TOC, file outline) inlined here",
    )
    structure_ref: str | None = Field(
        default=None,
        description="path to a sidecar JSON when the structure is too large to inline",
    )

    # Files that were found but not ingested because their type isn't supported in this milestone.
    ignored: list[str] = Field(
        default_factory=list,
        description="filenames found in this source but not ingested in the current milestone",
    )

    # Git-source metadata (only populated when location_kind == git).
    git_remote: str | None = None
    git_branch: str | None = None
    git_head_sha: str | None = None
    git_head_author: str | None = None
    git_head_email: str | None = None
    git_head_date: datetime | None = None
    git_head_subject: str | None = None
    git_dirty: bool | None = None
    git_remotes: dict[str, str] | None = None  # all remotes, parsed from `git remote -v`
    github_description: str | None = None  # only when origin is on GitHub

    # Obsidian-vault metadata (only populated when the directory is a vault).
    obsidian_vault_name: str | None = None
    obsidian_config: dict[str, object] | None = None

    claims: list[Claim] = Field(default_factory=list)


class SynthesisPage(PageBase):
    """Cross-source pages produced from accepted Cogitate proposals (Phase 2)."""

    type: Literal[PageType.SYNTHESIS] = PageType.SYNTHESIS  # type: ignore[assignment]
    sources: list[str] = Field(
        default_factory=list,
        description="ids of source pages this synthesis draws from",
    )
    claims: list[Claim] = Field(default_factory=list)


# ---- discriminated union for round-tripping ----


Page = Annotated[
    EntityPage | ConceptPage | SourcePage | SynthesisPage,
    Field(discriminator="type"),
]
"""Any Smalt page, discriminated on `type`."""
