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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- id validation ----
#
# Page ids and proposal ids become path components on disk
# (`pages/<subdir>/<id>.md`, `tasks/proposals/<subdir>/<id>.md`). The same is
# true for `ProposalPage.proposed_by` (the subdir). Reject anything that:
#
#   - would escape its target directory (`..`, `/`, `\`, leading `.`)
#   - is non-portable across Windows/macOS/Linux (`<>:"|?*`, whitespace,
#     control chars, leading dash/underscore, Windows-reserved filenames
#     like CON / NUL / COM1)
#   - is empty or longer than ~250 chars (most filesystems cap filenames
#     at 255 bytes; we leave headroom for the `.md` extension)
#
# The regex below enforces the structural rule; the reserved-name check
# catches the names that fit the regex but break on Windows.
#
# **Section ids** (the planned `<source-id>::<rel-path>` shape from M3
# ingest) are intentionally NOT permitted by this rule yet — they require
# a different validator that allows `::` and `/`. When section pages land
# the validator gets a second mode; for now everything must conform to
# this stricter slug-shape.

_PAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,253}$")

_WINDOWS_RESERVED: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _validate_id(value: str) -> str:
    """Validate a string that will become a path component (filename or subdir).

    Returns the value unchanged on success; raises ValueError with a clear
    message on failure (Pydantic surfaces this as the validation error).
    """
    if not _PAGE_ID_RE.match(value):
        raise ValueError(
            f"id must match {_PAGE_ID_RE.pattern!r}: alphanumeric start, "
            f"alphanumeric / underscore / hyphen body, 1..254 chars; got {value!r}"
        )
    if value.lower() in _WINDOWS_RESERVED:
        raise ValueError(
            f"id {value!r} is a Windows-reserved filename; pick a different slug"
        )
    return value

# ---- enums ----


class PageType(StrEnum):
    """Top-level kind of Smalt page. Each type has its own frontmatter shape."""

    ENTITY = "entity"
    CONCEPT = "concept"
    SOURCE = "source"
    SYNTHESIS = "synthesis"  # cross-source pages, written by Cogitate (Phase 2)


class ProposalKind(StrEnum):
    """Kind of proposal. Determines which downstream system / lifecycle applies.

    The full set is described in `cobalt-grinding/docs/plan.md` → "Proposal
    document shape and lifecycle"; keep this enum in sync as new kinds land.
    """

    # Schema layer — Cogitate proposes additions; Curate flags drift/removal.
    SCHEMA_ADDITION = "schema_addition"
    SCHEMA_DRIFT = "schema_drift"
    SCHEMA_REMOVAL = "schema_removal"

    # Graph structure — Cogitate constructs; Curate critiques.
    WIKI_EDGE = "wiki_edge"
    CONCEPT_MERGE = "concept_merge"
    NOVEL_CONCEPT = "novel_concept"
    CONTRADICTION = "contradiction"
    NOVEL_SYNTHESIS = "novel_synthesis"

    # Corpus growth — Research proposes new sources.
    SOURCE_ADOPTION = "source_adoption"

    # Capability surface — Toolsmith (Phase 3).
    TOOL_ADOPTION = "tool_adoption"
    TOOL_SPECIFICATION = "tool_specification"
    TOOLKIT_ADDITION = "toolkit_addition"
    TOOLKIT_REMOVAL = "toolkit_removal"

    # Curate audits.
    ORPHAN = "orphan"
    DUPLICATE = "duplicate"
    BROKEN_LINK = "broken_link"
    STALENESS = "staleness"


class ProposalStatus(StrEnum):
    """Lifecycle state of a proposal."""

    PROPOSED = "proposed"
    UNDER_TEST = "under_test"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPLIED = "applied"
    SUPERSEDED = "superseded"


class TestStatus(StrEnum):
    """Test outcome for a proposal's prediction."""

    UNTESTED = "untested"
    PASSED = "passed"
    FAILED = "failed"
    UNTESTABLE = "untestable"  # falsifiability gap; the user is the test


class TestCost(StrEnum):
    """Coarse cost tier. Governs whether the system auto-tests."""

    TRIVIAL = "trivial"   # no test required; user-approve and apply
    CHEAP = "cheap"       # auto-test
    MEDIUM = "medium"     # test if budget allows
    EXPENSIVE = "expensive"  # run only on user request


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


# ---- discriminated union for round-tripping ----


Page = Annotated[
    EntityPage | ConceptPage | SourcePage | SynthesisPage,
    Field(discriminator="type"),
]
"""Any Smalt page, discriminated on `type`."""


# ---- ProposalPage (lives outside the Page union) ----
#
# Proposals don't live in `pages/` — they live in `tasks/proposals/<system or
# schema>/`. The indexer walks `pages/` only, so ProposalPages aren't
# projected into the LanceDB pages table. Keeping them out of the discriminated
# `Page` union keeps that boundary clean.
#
# Per `cobalt-grinding/docs/plan.md` → "Proposal document shape and lifecycle":
# every system that doesn't directly write the corpus emits proposals (Curate,
# Cogitate, Research, Converse's novelty detector, Toolsmith). Each proposal
# is a hypothesis with a falsifiable prediction; the system tests where cheap
# and the user reviews hypothesis + evidence together.


class ProposalPage(BaseModel):
    """Structured proposal — Observation/Hypothesis/Prediction/Test in body;
    lifecycle + provenance metadata in frontmatter.

    The body is plain markdown with an expected section ordering
    (Observation, Hypothesis, Prediction, Test, Reasoning); this model only
    validates the frontmatter shape.
    """

    model_config = ConfigDict(extra="allow")  # forward-compat for new kinds/fields

    id: str = Field(description="stable proposal id; usually a slug")
    type: Literal["proposal"] = "proposal"
    title: str
    proposal_kind: ProposalKind
    status: ProposalStatus = ProposalStatus.PROPOSED
    proposed_by: str = Field(
        description=(
            "name of the agentic system that emitted this proposal — typically "
            "one of: cogitate, curate, research, converse, toolsmith"
        ),
    )
    proposed_at: datetime
    test_status: TestStatus = TestStatus.UNTESTED
    test_cost: TestCost = TestCost.MEDIUM
    related_pages: list[str] = Field(
        default_factory=list,
        description="ids of pages this proposal references (its `Observation` source material)",
    )
    supersedes: str | None = Field(
        default=None, description="proposal id this proposal supersedes, if any"
    )
    superseded_by: str | None = Field(
        default=None,
        description="proposal id that supersedes this one (set when a later proposal lands)",
    )

    # Both `id` and `proposed_by` become path components for the routing
    # in `tools._proposal_target_path` (`tasks/proposals/<proposed_by>/<id>.md`).
    # Apply the same path-traversal + portability guard as PageBase.id.
    @field_validator("id", "proposed_by")
    @classmethod
    def _check_path_components(cls, v: str) -> str:
        return _validate_id(v)
