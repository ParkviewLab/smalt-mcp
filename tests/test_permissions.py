# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""An unconfigured token means read-only (the handbook's mcp-server-conventions.md, "Auth model")."""

from __future__ import annotations

import pytest

from smalt_mcp.permissions import Scope, effective_scope, expected_internal_token, scope_for_token


@pytest.mark.parametrize("requested", list(Scope))
def test_no_token_caps_every_scope_at_read_only(requested: Scope) -> None:
    assert effective_scope(requested, None) is Scope.READ_ONLY


@pytest.mark.parametrize("requested", list(Scope))
def test_a_configured_token_keeps_the_requested_scope(requested: Scope) -> None:
    assert effective_scope(requested, "secret") is requested


def test_an_empty_token_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALT_INTERNAL_TOKEN", "")
    assert expected_internal_token() is None


def test_scope_for_token_without_a_configured_token_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMALT_INTERNAL_TOKEN", raising=False)
    assert scope_for_token(None) is Scope.READ_ONLY
    assert scope_for_token("anything") is Scope.READ_ONLY


def test_scope_for_token_with_a_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALT_INTERNAL_TOKEN", "secret")
    assert scope_for_token("secret") is Scope.READ_WRITE
    assert scope_for_token("wrong") is Scope.READ_ONLY
    assert scope_for_token(None) is Scope.READ_ONLY
