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
import time
from collections.abc import Iterator
from contextlib import contextmanager


class CorpusWriteMutex:
    """A named mutex for the single corpus-write critical section.

    Usage:
        with mutex.acquire("indexer"):
            # apply page writes + index updates here
            ...

    **Contention metrics (C-8).** The mutex tracks two counters used by
    the `/admin/health` route + `index_status` MCP tool:

    - `acquire_count` — total number of acquires since server start.
    - `total_wait_seconds` — cumulative time threads spent blocked on
      `self._lock.acquire()`. For a never-contended mutex this stays at 0.

    Both counters are server-lifetime totals (never reset). The derived
    `mean_wait_ms` property converts to a per-acquire mean — `0.0` if
    `acquire_count` is 0, otherwise `total_wait_seconds * 1000 /
    acquire_count`. v0 doesn't track per-tool breakdowns (we just sum
    across all holder names); per-tool histograms can come later if a
    real workload demands it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # The holder is read by status tooling (any thread, while the corpus
        # may or may not be locked) so it gets its own tiny lock. Otherwise
        # there's a race where `locked` is True but `holder` is None — small
        # but messy on a status display.
        self._holder_lock = threading.Lock()
        self._holder: str | None = None
        # Contention counters (C-8). Both protected by `_holder_lock` —
        # we already hold it on every acquire/release; piggy-backing is
        # cheaper than a third lock.
        self._acquire_count: int = 0
        self._total_wait_seconds: float = 0.0

    @contextmanager
    def acquire(self, holder_name: str) -> Iterator[None]:
        wait_start = time.perf_counter()
        self._lock.acquire()
        wait_seconds = time.perf_counter() - wait_start
        with self._holder_lock:
            self._holder = holder_name
            self._acquire_count += 1
            self._total_wait_seconds += wait_seconds
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

    @property
    def acquire_count(self) -> int:
        """Total acquires since server start (server-lifetime counter)."""
        with self._holder_lock:
            return self._acquire_count

    @property
    def total_wait_seconds(self) -> float:
        """Cumulative wait time across all acquires."""
        with self._holder_lock:
            return self._total_wait_seconds

    @property
    def mean_wait_ms(self) -> float:
        """Mean wait time per acquire, in ms. 0.0 if no acquires yet."""
        with self._holder_lock:
            if self._acquire_count == 0:
                return 0.0
            return (self._total_wait_seconds / self._acquire_count) * 1000.0
