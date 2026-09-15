"""Static file serving that can bring a job's files back before answering 404.

``OUTPUT_DIR`` is not durable: a redeploy or the hourly sweep wipes a
finished job, while the user's library still points at
``/videos/<job_id>/<file>``. The API endpoints re-pull a project from R2 on
demand (``app._ensure_job_files``), but ``/videos`` was a plain
``StaticFiles`` mount, so when the dashboard reopened a project it fired the
15 transcript requests (which restored the job, ~25 s for 2 GB) and the 15
``<video>`` loads at the same instant, and every player got a 404 that only
a reload fixed (job cff3ad6c, 6-sep-2026 00:37 UTC).

This subclass keeps everything StaticFiles does (Range requests, ETags, HEAD)
and adds one step: on a miss it hands the first path segment (the job id) to
``restorer``; when that reports the files may now be there, it looks once
more. The restorer is awaited on the request, so a player that arrives while
a restore is in flight simply waits for it (the per-job lock lives in the
restorer) instead of failing.
"""
from typing import Awaitable, Callable, Optional

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

Restorer = Callable[[str], Awaitable[bool]]


class RestoringStaticFiles(StaticFiles):
    def __init__(self, *args, restorer: Optional[Restorer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.restorer = restorer

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or self.restorer is None:
                raise
            job_id = path.split("/", 1)[0] if "/" in path else ""
            if not job_id:
                raise
            try:
                restored = await self.restorer(job_id)
            except Exception:
                restored = False
            if not restored:
                raise
            return await super().get_response(path, scope)
