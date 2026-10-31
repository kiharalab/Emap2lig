"""Single-worker job queue for GPU-bound pipeline stages.

Detection and modeling jobs share one asyncio worker so two runs cannot contend
for the same accelerator. Cooperative cancel is handled on the :class:`Job`:
queued jobs are dropped; running jobs set ``cancel_event`` and the progress
callback raises :class:`~emap2lig.errors.JobCancelledError` at the next event.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from .state import Job, get_job

logger = logging.getLogger(__name__)

RunFn = Callable[[], Awaitable[None]]


class JobRunner:
    """FIFO queue with a single consumer task."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._runs: dict[str, RunFn] = {}
        self._queued: list[str] = []
        self._cancelled: set[str] = set()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the consumer task. Idempotent."""
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._consume(), name="job-runner")

    async def stop(self) -> None:
        """Cancel the consumer task."""
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    def submit(self, job: Job, run: RunFn) -> None:
        """Enqueue *job*. Status becomes ``queued``.

        Args:
            job: Newly created job to run.
            run: Async callable that executes the pipeline stage.
        """
        job.status = "queued"
        self._runs[job.id] = run
        self._queued.append(job.id)
        self._queue.put_nowait(job.id)

    def queue_position(self, job_id: str) -> int | None:
        """1-based position among still-queued jobs, or ``None``.

        Args:
            job_id: Job to look up.

        Returns:
            Queue position starting at 1, or ``None`` if the job is not queued.
        """
        try:
            return self._queued.index(job_id) + 1
        except ValueError:
            return None

    def cancel(self, job: Job) -> bool:
        """Cancel a queued or running job.

        Args:
            job: Job to cancel.

        Returns:
            ``True`` if the job was queued or running and cancel was accepted.
        """
        if job.status == "queued":
            self._cancelled.add(job.id)
            if job.id in self._queued:
                self._queued.remove(job.id)
            self._runs.pop(job.id, None)
            job.status = "cancelled"
            job.progress = "Cancelled"
            job.finished_at = datetime.now(UTC)
            return True
        if job.status == "running":
            job.cancel_event.set()
            return True
        return False

    async def _consume(self) -> None:
        """Pop jobs from the queue and run them one at a time."""
        while True:
            job_id = await self._queue.get()
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                self._runs.pop(job_id, None)
                continue
            run = self._runs.pop(job_id, None)
            if job_id in self._queued:
                self._queued.remove(job_id)
            if run is None:
                continue
            job = get_job(job_id)
            if job is None or job.status == "cancelled":
                continue
            job.status = "running"
            job.started_at = datetime.now(UTC)
            try:
                await run()
            except Exception:
                logger.exception("Job %s crashed outside the stage handler", job_id)
                if job.status not in {"completed", "cancelled", "failed"}:
                    job.status = "failed"
                    job.error = "internal error"
                    job.finished_at = datetime.now(UTC)
