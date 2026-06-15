# SPDX-FileCopyrightText: 2026 Gary Frattarola <garyf@parkviewlab.ai>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Resolve deps first so they cache across source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# README.md is part of the package metadata (pyproject.toml -> readme).
# uv sync --no-install-project skipped reading it; the second sync
# (which installs the project itself) does, so it must be present.
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

ENV PYTHONUNBUFFERED=1 \
    SMALT_DIR=/data \
    PORT=35833 \
    HOST=0.0.0.0

EXPOSE 35833
VOLUME ["/data"]

CMD ["uv", "run", "python", "-m", "smalt_mcp"]
