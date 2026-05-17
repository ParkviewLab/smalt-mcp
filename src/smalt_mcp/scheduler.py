"""In-process async task scheduler for long-running operations (C-13).

Heavy operations (`reindex_all`, future large batch writes) enqueue an
async task via the `Scheduler`, get back a `task_id`, and run in the
background via `asyncio.Task`. Clients poll `task_status` /
`task_list` to track state and request cancellation via `task_cancel`.

Lifecycle
---------

    pending → running → (succeeded | failed | cancelled)

`cancelled` can also be reached from `pending` if cancel arrives
before the task is scheduled to run.

Cancellation is cooperative: long-running synchronous work (the
indexer) runs via `asyncio.to_thread()` and the wrapper checks
`task.cancel_requested` at safe boundaries. Hard mid-LanceDB-write
cancellation would leave the index in a half-state and is intentionally
NOT supported in v1.0 — the worst-case latency between a cancel
request and the task actually stopping is "one file's worth of indexer
work" (typically <1s).

Persistence
-----------

In-memory only. Tasks live for the process lifetime; completed tasks
get garbage-collected `gc_ttl_seconds` after they finish (default 1
hour — long enough for an operator to poll a result later, short
enough that a long-lived server doesn't accumulate gigabytes of stale
task records). Persistence across restarts is out of scope for v1.0;
operators should not restart mid-task.

Thread safety
-------------

All public Scheduler methods must be called from the asyncio event
loop thread. The work functions themselves are free to dispatch
blocking work to threads via `asyncio.to_thread()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Set of states that mean "task is done, won't change further."
TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED}
)


# How long a completed task stays in the registry before GC. Default
# 1h is long enough for a human operator to poll a result later, short
# enough that a long-lived server doesn't accumulate stale records.
DEFAULT_GC_TTL_SECONDS = 3600

# GC interval. Runs once a minute by default; nothing here is
# time-critical, just bounds memory.
_GC_INTERVAL_SECONDS = 60


WorkFn = Callable[["Task"], Awaitable[Any]]


@dataclass
class Task:
    """One scheduled task.

    Attributes
    ----------
    id: short URL-safe random string identifying the task.
    kind: caller-supplied label for the operation kind
        (e.g. ``"reindex_all"``).
    state: current lifecycle state.
    created_at / started_at / finished_at: timestamps. `started_at` is
        None while pending; `finished_at` is None until terminal.
    progress: opaque dict the work function may write to; surfaced
        verbatim by `task_status`. Schema is per-kind; the scheduler
        doesn't interpret it.
    result: the success payload (whatever the work function returned).
        None unless state is SUCCEEDED.
    error: short error string (`"<ExceptionType>: <message>"`). None
        unless state is FAILED.
    cancel_requested: set by `Scheduler.cancel()`. The work function
        is responsible for checking it at safe boundaries via
        `check_cancel()`.
    """

    id: str
    kind: str
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    cancel_requested: bool = False
    # The underlying asyncio.Task. Stored so the scheduler can cancel
    # it on shutdown / explicit cancel. repr=False keeps the dataclass
    # repr usable in logs.
    _asyncio_task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def check_cancel(self) -> None:
        """Raise `asyncio.CancelledError` if `cancel_requested` is set.

        Work functions should call this at boundaries where it's safe
        to abort — between indexer iterations, between batched lance
        writes, etc.
        """
        if self.cancel_requested:
            raise asyncio.CancelledError(f"task {self.id} cancelled")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the task — what `task_status` returns."""
        return {
            "task_id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": dict(self.progress),
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class Scheduler:
    """In-process scheduler.

    See module docstring for lifecycle / threading semantics. Typical
    usage is one Scheduler per `App` instance, started in the FastAPI
    lifespan and shut down on app exit.
    """

    def __init__(self, *, gc_ttl_seconds: int = DEFAULT_GC_TTL_SECONDS) -> None:
        self._tasks: dict[str, Task] = {}
        self._gc_ttl_seconds = gc_ttl_seconds
        self._gc_task: asyncio.Task[None] | None = None

    # ---- enqueue / inspect ----

    def enqueue(self, kind: str, work: WorkFn) -> Task:
        """Create a Task and schedule `work(task)` on the running loop.

        Returns immediately with the Task in `PENDING` state — the
        actual `_runner` coroutine moves it to `RUNNING` the first
        time the event loop awaits it.

        Requires a running asyncio event loop (raises `RuntimeError`
        otherwise). Safe to call from any handler / lifespan code that
        already runs inside the loop.
        """
        task = Task(id=_new_task_id(), kind=kind)
        self._tasks[task.id] = task
        task._asyncio_task = asyncio.create_task(
            self._runner(task, work), name=f"task-{task.id}"
        )
        return task

    async def _runner(self, task: Task, work: WorkFn) -> None:
        """Drive the lifecycle for one task.

        Wraps `work(task)` with state transitions + exception capture
        so failures don't escape to the loop's default error handler.
        """
        task.state = TaskState.RUNNING
        task.started_at = datetime.now(UTC)
        try:
            result = await work(task)
            # `work` may have ignored cancel_requested; honor it on
            # exit so the caller can rely on cancel being authoritative.
            if task.cancel_requested:
                task.state = TaskState.CANCELLED
            else:
                task.state = TaskState.SUCCEEDED
                task.result = result
        except asyncio.CancelledError:
            task.state = TaskState.CANCELLED
            # Don't re-raise — let the asyncio task end cleanly so the
            # event loop doesn't log a CancelledError traceback.
        except Exception as e:  # noqa: BLE001 — capture every failure
            task.state = TaskState.FAILED
            task.error = f"{type(e).__name__}: {e}"
            logger.exception("scheduled task %s (%s) failed", task.id, task.kind)
        finally:
            task.finished_at = datetime.now(UTC)

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(
        self,
        *,
        state: TaskState | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks, most-recent first, optionally filtered.

        `limit` is applied AFTER filtering — so `state=SUCCEEDED,
        limit=10` returns the 10 most recent successful tasks, not "10
        most recent tasks of which some happen to be successful."
        """
        tasks = list(self._tasks.values())
        if state is not None:
            tasks = [t for t in tasks if t.state == state]
        if kind is not None:
            tasks = [t for t in tasks if t.kind == kind]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    # ---- cancel ----

    def cancel(self, task_id: str) -> Task | None:
        """Request cancellation. Returns the task (or None if unknown).

        - PENDING task → transitions straight to CANCELLED; the
          underlying asyncio.Task is also cancelled so it never runs.
        - RUNNING task → sets `cancel_requested`. The work function
          must call `task.check_cancel()` at a safe boundary to honor
          it; if it doesn't, the task continues and may still complete
          normally (in which case the runner's exit-time check picks up
          the cancel flag and marks it CANCELLED anyway). The
          asyncio.Task is also cancelled so any pending await unblocks.
        - Terminal task → no-op; returns the existing task.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.state in TERMINAL_STATES:
            return task
        task.cancel_requested = True
        if task.state == TaskState.PENDING:
            task.state = TaskState.CANCELLED
            task.finished_at = datetime.now(UTC)
        if task._asyncio_task is not None and not task._asyncio_task.done():
            task._asyncio_task.cancel()
        return task

    # ---- GC loop ----

    def start_gc(self) -> None:
        """Start the periodic GC task. Idempotent — safe to call twice."""
        if self._gc_task is None or self._gc_task.done():
            self._gc_task = asyncio.create_task(self._gc_loop(), name="scheduler-gc")

    async def stop_gc(self) -> None:
        """Stop the GC task. Awaits its completion so the lifespan exit
        doesn't leak a running task."""
        if self._gc_task is not None:
            self._gc_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gc_task
            self._gc_task = None

    async def shutdown(self) -> None:
        """Cancel every running task and stop the GC loop.

        Called from the FastAPI lifespan exit so the server doesn't
        leak background tasks. Best-effort — tasks that ignore
        cancellation might still be in-flight when this returns; the
        loop teardown will eventually take them down.
        """
        for task in list(self._tasks.values()):
            if task.state in TERMINAL_STATES:
                continue
            task.cancel_requested = True
            if task._asyncio_task is not None and not task._asyncio_task.done():
                task._asyncio_task.cancel()
        # Give cancelled tasks a tick to actually finish.
        running = [
            t._asyncio_task
            for t in self._tasks.values()
            if t._asyncio_task is not None and not t._asyncio_task.done()
        ]
        for at in running:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await at
        await self.stop_gc()

    async def _gc_loop(self) -> None:
        """Periodically purge terminal tasks older than `gc_ttl_seconds`."""
        try:
            while True:
                await asyncio.sleep(_GC_INTERVAL_SECONDS)
                self._gc_once()
        except asyncio.CancelledError:
            return

    def _gc_once(self) -> int:
        """Remove terminal tasks older than the TTL. Returns count removed."""
        now = datetime.now(UTC)
        stale = [
            tid
            for tid, t in self._tasks.items()
            if t.state in TERMINAL_STATES
            and t.finished_at is not None
            and (now - t.finished_at).total_seconds() > self._gc_ttl_seconds
        ]
        for tid in stale:
            del self._tasks[tid]
        return len(stale)


def _new_task_id() -> str:
    """Short random task id — 12 base32-ish chars, ~72 bits of entropy.

    Short enough to type / scan in logs; long enough that collisions
    don't matter in practice (need ~2^36 tasks for a 1% birthday
    chance, far past any plausible task volume).
    """
    return secrets.token_urlsafe(9)
