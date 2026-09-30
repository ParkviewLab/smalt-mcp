# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Read-only vs. read-write scope filter.

Tools are tagged with a `Scope` at registration time. The server reads
`SMALT_SCOPE` once at startup and uses that single server-wide scope when
listing or dispatching tools; a caller at tier N may see and call any tool
whose required scope is at most N.

An unconfigured token means read-only (the handbook's
mcp-server-conventions.md, "Auth model"): while `SMALT_INTERNAL_TOKEN` is
unset, the server's scope is capped at `READ_ONLY` whatever `SMALT_SCOPE`
asks for, so the default an operator gets without thinking about it is the
safe one. With the token set, `SMALT_SCOPE` applies as configured. The token
is not yet checked on incoming requests; no server in the ParkviewLab family
does that yet. `scope_for_token` is the per-client mapping that such a check
would use.
"""

from __future__ import annotations

import os
from enum import StrEnum


class Scope(StrEnum):
    """Tiered permission scope.

    Tier order: READ_ONLY (0) < READ_WRITE (1) < REMOVE_DESTRUCTIVE (2).
    A caller at tier N may see and call any tool whose required scope is ≤ N.
    """

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    REMOVE_DESTRUCTIVE = "remove_destructive"


# Numeric tier per scope; used for the inclusion check (caller_tier >= tool_tier).
SCOPE_TIER: dict[Scope, int] = {
    Scope.READ_ONLY: 0,
    Scope.READ_WRITE: 1,
    Scope.REMOVE_DESTRUCTIVE: 2,
}


# Single shared secret. An empty value counts as unset.
_INTERNAL_TOKEN_ENV = "SMALT_INTERNAL_TOKEN"


def expected_internal_token() -> str | None:
    """Return the configured internal token, or None if it is not set."""
    return os.environ.get(_INTERNAL_TOKEN_ENV) or None


def effective_scope(requested: Scope, token: str | None) -> Scope:
    """The server-wide scope: `requested`, capped at READ_ONLY when no token is configured."""
    if token is None:
        return Scope.READ_ONLY
    return requested


def scope_for_token(presented: str | None) -> Scope:
    """Map a presented token to a scope.

    No token configured: every client is READ_ONLY, since an unconfigured
    token means read-only. Configured: only callers presenting the matching
    token get READ_WRITE; everyone else is READ_ONLY.
    """
    expected = expected_internal_token()
    if expected is None:
        return Scope.READ_ONLY
    if presented and presented == expected:
        return Scope.READ_WRITE
    return Scope.READ_ONLY
