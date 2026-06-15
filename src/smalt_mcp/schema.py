# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

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

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- id validation ----
#
# Two id shapes are accepted:
#
# 1. **Slug ids** (entities, concepts, syntheses, source-root pages,
#    index pages): alphanumeric start, alphanumeric/underscore/hyphen
#    body, 1..254 chars. Become a single path component on disk
#    (`pages/<subdir>/<id>.md`).
#
# 2. **Section ids** (M3 hybrid layout): `<source-id>::<rel-path>`. The
#    source-id part is a slug id; the rel-path part is one or more
#    /-separated path components. Become a nested path on disk
#    (`pages/sources/<source-id>/<rel-path>.md` — the `::` translates
#    to `/`). Section ids are the M3 hybrid-layout case where a
#    multi-file source has one page per file. The id encodes the
#    file's location within the source, which keeps the canonical id
#    stable across renames at the source-root level.
#
# In both shapes we reject anything that:
#   - would escape its target directory (`..`, leading `/`, leading `.`)
#   - is non-portable across Windows/macOS/Linux (`<>:"|?*`, whitespace,
#     control chars, Windows-reserved filenames like CON / NUL / COM1)
#   - is empty or longer than 254 chars total (most filesystems cap
#     filenames at 255 bytes; we leave headroom for the `.md` extension)
#
# Each path component in a section id is itself validated against the
# (relaxed) per-component regex below — alphanumeric / underscore /
# hyphen / dot, must start with alphanumeric. The leading-alphanumeric
# rule means hidden files (.foo) are rejected; the dot-in-body allowance
# lets real filenames like `utils.py` or `package.json` through.

_PAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,253}$")
"""Slug id shape — used directly for non-section ids; used for the
source-id portion of section ids."""

_SECTION_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
"""Per-component shape for the rel-path portion of a section id.
Same alphabet as slug ids plus dots — for real filename extensions."""

_WINDOWS_RESERVED: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)


def _validate_id(value: str) -> str:
    """Validate a page id. Routes by shape:

    - Contains `::` → section id (`<source-id>::<rel-path>`) → validated
      via `_validate_section_id`.
    - Otherwise → slug id → validated via `_validate_slug_id`.

    Returns the value unchanged on success; raises ValueError with a
    clear message on failure (Pydantic surfaces this as the validation
    error).
    """
    if "::" in value:
        return _validate_section_id(value)
    return _validate_slug_id(value)


def _validate_slug_id(value: str) -> str:
    """Standard slug-shape id — used directly for non-section pages and
    for the source-id portion of section ids."""
    if not _PAGE_ID_RE.match(value):
        raise ValueError(
            f"id must match {_PAGE_ID_RE.pattern!r}: alphanumeric start, "
            f"alphanumeric / underscore / hyphen body, 1..254 chars; got {value!r}"
        )
    if value.lower() in _WINDOWS_RESERVED:
        raise ValueError(f"id {value!r} is a Windows-reserved filename; pick a different slug")
    return value


def _validate_section_id(value: str) -> str:
    """Validate a section page id: `<source-id>::<rel-path>`.

    The rel-path may contain `/` separating components. Each component
    is alphanumeric+`._-` with an alphanumeric leading char (no hidden
    files via `.foo`; no path-escape via `..`; no absolute-path escape
    via leading `/`).
    """
    if len(value) > 254:
        raise ValueError(
            f"section id too long ({len(value)} chars, max 254 — leaves "
            f"headroom for the .md extension on disk); got {value!r}"
        )
    if value.count("::") != 1:
        raise ValueError(f"section id must contain exactly one '::' separator; got {value!r}")

    source_id, rel_path = value.split("::", 1)

    # Source-id part: standard slug.
    try:
        _validate_slug_id(source_id)
    except ValueError as e:
        raise ValueError(f"section id source-id part {source_id!r} is invalid: {e}") from e

    # Rel-path part: non-empty, no leading slash, components valid.
    if not rel_path:
        raise ValueError(f"section id rel-path (after '::') must be non-empty; got {value!r}")
    if rel_path.startswith("/"):
        raise ValueError(
            f"section id rel-path must not start with '/' (absolute paths "
            f"would escape the source directory); got {value!r}"
        )

    components = rel_path.split("/")
    for comp in components:
        if not comp:
            raise ValueError(f"section id rel-path must not contain '//' (empty component); got {value!r}")
        if comp in (".", ".."):
            raise ValueError(f"section id rel-path must not contain '{comp}' (path traversal); got {value!r}")
        if not _SECTION_PATH_COMPONENT_RE.match(comp):
            raise ValueError(
                f"section id rel-path component {comp!r} contains disallowed "
                f"characters; allowed: alphanumeric / underscore / hyphen / dot, "
                f"must start with alphanumeric; got {value!r}"
            )
        # Windows-reserved check on component base (filename without extension).
        # `con.py` is rejected because `con` (the base) is reserved on Windows.
        base = comp.split(".", 1)[0]
        if base.lower() in _WINDOWS_RESERVED:
            raise ValueError(
                f"section id rel-path component {comp!r} starts with "
                f"Windows-reserved filename {base!r}; pick a different path"
            )

    return value


# ---- enums ----


class PageType(StrEnum):
    """Top-level kind of Smalt page. Each type has its own frontmatter shape."""

    ENTITY = "entity"
    CONCEPT = "concept"
    SOURCE = "source"
    SYNTHESIS = "synthesis"  # cross-source pages, written by Cogitate (Phase 2)
    INDEX = "index"  # auto-generated index pages (glossary, domains, ...); see IndexPage


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


class Evidence(BaseModel):
    """A `(source_id, snippet)` pair attached to a concept, recording *where* a
    term was used in a particular source. Glossary entries on `ConceptPage`
    accumulate evidence across sources so a reader (or an agent) can see the
    in-context uses without re-fetching the source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="id of the SourcePage this evidence came from")
    snippet: str = Field(description="short verbatim excerpt; provenance, not a copy of the source")


# ---- common base for all page types ----


class PageBase(BaseModel):
    """Fields shared by every page type."""

    model_config = ConfigDict(extra="allow")  # allow forward-compat keys we don't model yet

    id: str = Field(description="stable Smalt-internal id; usually a slug")
    type: PageType
    title: str

    # Path-traversal + portability guard. Applied to every page type via
    # inheritance. See `_validate_id` for the rule.
    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return _validate_id(v)

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
    domains: list[str] = Field(
        default_factory=list,
        description=(
            "ids of ConceptPages this entity belongs to. Used for disambiguation "
            "when same-name entities live in different domains (e.g. a CS Smith "
            "vs. an economist Smith)."
        ),
    )
    claims: list[Claim] = Field(default_factory=list)


class ConceptPage(PageBase):
    """A concept page. Three notable flags:

    - `is_domain=True`: this concept is itself a domain. The auto-generated
      `pages/domains.md` IndexPage lists all of these. Domain hierarchy is
      expressed by `subdomain_of` labeled links (in `links_out`), NOT by the
      `domains:` field — keeps "what this is about" cleanly separated from
      "what this is under".
    - `glossary=True`: this is a short-definition glossary entry. The
      `evidence` list collects per-source snippets so a reader can see where
      the term was used. Richer concepts (parents, claims) leave the flag
      false.
    - `domains: [...]`: ids of ConceptPages (themselves marked `is_domain`)
      that this concept is *about*. Multi-domain by default — a concept can
      belong to several domains.
    """

    type: Literal[PageType.CONCEPT] = PageType.CONCEPT  # type: ignore[assignment]
    parents: list[str] = Field(
        default_factory=list,
        description="ids of broader concept pages (taxonomy edges, populated by Cogitate)",
    )
    domains: list[str] = Field(
        default_factory=list,
        description=(
            "ids of ConceptPages this concept belongs to. Use `subdomain_of` "
            "labeled links for hierarchy between domains themselves."
        ),
    )
    is_domain: bool = Field(
        default=False,
        description=(
            "When true, this concept is itself a domain. Listed in the "
            "auto-generated pages/domains.md IndexPage."
        ),
    )
    glossary: bool = Field(
        default=False,
        description=(
            "When true, this is a short-definition glossary entry; listed in "
            "the auto-generated pages/glossary.md IndexPage."
        ),
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Per-source (source_id, snippet) pairs; used by glossary entries.",
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

    # Domain hints — multi-domain by default. Gives the SME ingest agent a
    # default for terms extracted from this source.
    domains: list[str] = Field(
        default_factory=list,
        description="ids of ConceptPages this source is about",
    )

    # Hybrid source layout — multi-file sources have an index page + one
    # section page per file; section pages link back via parent_source; the
    # index page enumerates its sections via `sections`.
    parent_source: str | None = Field(
        default=None,
        description="id of the parent SourcePage if this is a section page",
    )
    sections: list[str] = Field(
        default_factory=list,
        description="ids of section SourcePages this source contains",
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


class IndexPage(PageBase):
    """An auto-generated index page.

    The indexer rewrites the body of every IndexPage at the end of every
    run by executing the `stored_query` against the current corpus state.
    Humans/agents should NOT hand-edit the body — it'll be regenerated on
    the next indexer pass. To customize an index, change its
    `stored_query` (manually on disk for now; a tool-driven path may
    arrive later).

    Bootstrap creates two canonical IndexPages:
      - `pages/glossary.md` (id `idx-glossary`) — over `glossary: true`
        ConceptPages.
      - `pages/domains.md` (id `idx-domains`) — over `is_domain: true`
        ConceptPages.

    Future: `pages/entities.md`, `pages/sources.md`, and custom
    user-defined queries (with `auto_generated: true` always set).

    Direct writes to IndexPages via `write_page` / `write_pages` are
    rejected (`forbidden_page_type`); the indexer is the only writer.
    Note though that `remove_page` IS still allowed — useful for retiring
    a no-longer-wanted custom IndexPage. The two canonical IndexPages
    will be regenerated on the next bootstrap.
    """

    type: Literal[PageType.INDEX] = PageType.INDEX  # type: ignore[assignment]
    auto_generated: Literal[True] = True
    stored_query: dict[str, Any] = Field(
        description=(
            "Query that defines this index's contents. Initial shape: "
            "`{'kind': 'concept_flag', 'flag': 'glossary' | 'is_domain'}`. "
            "Future kinds (e.g. tag-set, type-and-domain) can be added "
            "non-breakingly — unknown kinds are ignored at regeneration "
            "time."
        ),
    )


# ---- discriminated union for round-tripping ----


Page = Annotated[
    EntityPage | ConceptPage | SourcePage | SynthesisPage | IndexPage,
    Field(discriminator="type"),
]
"""Any Smalt page, discriminated on `type`."""

# Proposals (the scientific-method surface) live in a separate substrate:
# the `ebony-enriching` MCP server, with its own `ProposalPage` schema and
# `EBONY_ENRICHING_DIR` storage. Smalt-mcp is purely the canonical
# knowledge substrate — no proposal / experiment / gap models live here.
