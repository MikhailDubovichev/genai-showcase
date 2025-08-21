"""
APScheduler-based daily scheduler for feedback synchronization.

This module provides a small, purpose-built wrapper around APScheduler to run the
edge feedback sync job once per day. APScheduler stands for Advanced Python Scheduler
(A = Advanced, P = Python, Scheduler), and we use its asyncio scheduler variant so it
plays nicely with FastAPI's event loop. The job is triggered with a daily cron trigger
at 02:00 local time. The term "cron" originates from the Greek "chronos" (time), and in
computing it refers to time-based job scheduling. We log a concise summary after each
run and ensure that failures never crash the application. The business logic for reading
local feedback, normalizing items, filtering by the checkpoint, and posting batches to
the Cloud remains in services.feedback_sync; this module focuses only on scheduling and
lifecycle wiring (start/stop) aligned with the app startup and shutdown events.
"""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .feedback_sync import run_feedback_sync


def run_sync_job(logger: logging.Logger) -> None:
    """
    Execute one feedback sync and log a concise summary; never raise exceptions.

    This function wraps `run_feedback_sync()` with defensive error handling so that
    scheduler invocations cannot crash the web application. On success, it logs the
    returned summary dictionary at INFO level. If any exception is raised by the
    underlying sync logic (e.g., network errors, configuration issues), the exception
    is caught and logged at WARNING level, and the function returns without re-raising.
    The scheduler will continue to operate and trigger future runs.
    """
    try:
        summary = run_feedback_sync()
        logger.info(
            "feedback_sync summary: sent=%s accepted=%s duplicates=%s skipped=%s",
            summary.get("sent", 0),
            summary.get("accepted", 0),
            summary.get("duplicates", 0),
            summary.get("skipped", 0),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("feedback_sync failed: %s", exc)


def start_feedback_scheduler(app) -> None:
    """
    Start the daily feedback sync scheduler and store it on the app state.

    Creates an `AsyncIOScheduler`, configures a `CronTrigger(hour=2, minute=0)` to run the
    job daily at 02:00 local time, and starts the scheduler. The job function is provided
    a module logger so it can emit a single-line summary per run and warnings on failure.
    The scheduler instance is attached to `app.state.feedback_scheduler` for later shutdown.
    """
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=2, minute=0)
    logger = logging.getLogger(__name__)
    scheduler.add_job(
        run_sync_job,
        trigger=trigger,
        args=[logger],
        id="feedback_sync_daily",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()
    setattr(app.state, "feedback_scheduler", scheduler)


def shutdown_feedback_scheduler(app) -> None:
    """
    Stop the feedback sync scheduler if it was started; swallow exceptions.

    Retrieves the scheduler stored on `app.state.feedback_scheduler`, calls `shutdown` with
    `wait=False` to request a non-blocking stop, and ignores any exceptions raised during
    shutdown so that application teardown remains robust.
    """
    scheduler = getattr(app.state, "feedback_scheduler", None)
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
    except Exception:  # pragma: no cover - defensive
        return


