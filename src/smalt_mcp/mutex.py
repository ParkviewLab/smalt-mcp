"""Single-writer corpus mutex.

smalt-mcp's invariant: only one task may be in the *commit* phase of a
corpus write at a time. The expensive parts of an ingestion (LLM calls,
fastembed inference, parsing) parallelize freely; only the atomic page-write
+ index-update step serializes.

The mutex wraps the entire `Indexer.run` — both its reads
(`existing_page_hashes`, `fetch_created_at`) and its writes (upserts, deletes,
FTS / ANN refresh). So two concurrent indexer runs fully serialize.

**Why this is safe even when reads escape the mutex:** LanceDB uses
snapshot-versioned reads. A reader sees the snapshot it opened — a
concurrent writer's commit becomes visible only after the reader finishes.
Worst case is doing redundant work (skipping a page that *just* changed)
— never corruption.

Implementation: a plain `threading.Lock` wrapped in a context manager
with a name so traces and logs make it obvious what's serializing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class CorpusWriteMutex:
    """A named mutex for the single corpus-write critical section.

    Usage:
        with mutex.acquire("indexer"):
            # apply page writes + index updates here
            ...
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # The holder is read by status tooling (any thread, while the corpus
        # may or may not be locked) so it gets its own tiny lock. Otherwise
        # there's a race where `locked` is True but `holder` is None — small
        # but messy on a status display.
        self._holder_lock = threading.Lock()
        self._holder: str | None = None

    @contextmanager
    def acquire(self, holder_name: str) -> Iterator[None]:
        self._lock.acquire()
        with self._holder_lock:
            self._holder = holder_name
        try:
            yield
        finally:
            with self._holder_lock:
                self._holder = None
            self._lock.release()

    @property
    def holder(self) -> str | None:
        """Return the name of the current holder, or None if unheld."""
        with self._holder_lock:
            return self._holder

    @property
    def locked(self) -> bool:
        return self._lock.locked()
