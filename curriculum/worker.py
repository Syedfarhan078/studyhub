"""
In-process and standalone background worker engine for StudyHub AI curriculum generation.

Provides:
- GenerationWorkerEngine: Shared worker logic (claiming with SKIP LOCKED, fencing,
  heartbeat renewal, lease recovery, execution, and error classification).
- JobHeartbeat: Context-managed daemon thread renewing leases during long Gemini API calls.
- InProcessWorkerThread: Managed background daemon thread running inside Gunicorn web process.
- start_in_process_worker() / stop_in_process_worker(): Thread lifecycle management.
- notify_new_job(): Thread-safe instant wake-up trigger for newly queued jobs.
"""

import os
import sys
import time
import uuid
import logging
import threading
from datetime import timedelta

from django.db import transaction, close_old_connections, DatabaseError
from django.utils import timezone
from django.conf import settings

from curriculum.models import GenerationJob, Subject
from curriculum.services.ai_curriculum import (
    generate_curriculum_from_prompt,
    generate_curriculum_data,
    persist_curriculum_records,
    CurriculumFencingError,
    CurriculumGenerationError,
    GeminiTransientError,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    CurriculumParseError,
    CurriculumValidationError,
)

logger = logging.getLogger(__name__)

# Module-level thread coordination primitives
_worker_lock = threading.RLock()
_worker_thread = None
_shutdown_event = threading.Event()
_job_posted_event = threading.Event()


def _safe_close_conns():
    """Recycles old database connections unless running in a Django test environment."""
    if 'test' in sys.argv or getattr(settings, 'TESTING', False):
        return
    try:
        close_old_connections()
    except Exception:
        pass


class JobHeartbeat:
    """
    Periodically refreshes GenerationJob.locked_at during execution while verifying
    that the job remains in RUNNING status and that the claim_token has not been revoked.
    Runs safely in a daemon thread with robust database connection lifecycle handling.
    """
    def __init__(self, job_id, claim_token, interval_seconds, worker_id):
        self.job_id = job_id
        self.claim_token = claim_token
        self.interval = interval_seconds
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread = None

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                try:
                    _safe_close_conns()
                    rows = GenerationJob.objects.filter(
                        id=self.job_id,
                        claim_token=self.claim_token,
                        status=GenerationJob.Status.RUNNING
                    ).update(locked_at=timezone.now())

                    if rows == 0:
                        logger.warning(
                            "[%s] Heartbeat failed for job %s: claim token %s revoked or expired.",
                            self.worker_id, self.job_id, self.claim_token
                        )
                        self.lost_event.set()
                        break
                    else:
                        logger.debug(
                            "[%s] Heartbeat refreshed lease for job %s (token %s).",
                            self.worker_id, self.job_id, self.claim_token
                        )
                except Exception as e:
                    logger.warning(
                        "[%s] Heartbeat encountered exception for job %s: %s",
                        self.worker_id, self.job_id, e
                    )
                finally:
                    _safe_close_conns()
        finally:
            try:
                from django.db import connection
                connection.close()
            except Exception:
                pass

    def __enter__(self):
        if self.interval > 0:
            self.thread = threading.Thread(
                target=self._run,
                name=f"heartbeat-{self.job_id}",
                daemon=True
            )
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        _safe_close_conns()


class GenerationWorkerEngine:
    """
    Core queueing and execution engine for GenerationJob processing.
    Shared between standalone management command and in-process background worker thread.
    """
    def __init__(
        self,
        worker_id: str = None,
        poll_interval: float = None,
        lease_duration: int = None,
        heartbeat_interval: int = None,
        max_attempts: int = None,
    ):
        self.worker_id = worker_id or f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.poll_interval = (
            poll_interval
            if poll_interval is not None
            else getattr(settings, 'GENERATION_WORKER_POLL_INTERVAL', 2.0)
        )
        self.lease_duration = (
            lease_duration
            if lease_duration is not None
            else getattr(settings, 'GENERATION_WORKER_LEASE_DURATION', 300)
        )
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else getattr(settings, 'GENERATION_WORKER_HEARTBEAT_INTERVAL', 30)
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else getattr(settings, 'GENERATION_WORKER_MAX_ATTEMPTS', 3)
        )

    def close_db_connections(self):
        """Close old database connections in worker daemon mode to prevent stale sockets."""
        _safe_close_conns()

    def reap_abandoned_jobs(self, lease_duration_seconds: int = None, max_attempts: int = None) -> int:
        """
        Detects jobs stuck in RUNNING status whose worker lease has expired.
        Requeues them if retries remain; marks them FAILED if attempts exhausted.
        Clears stale claim tokens to prevent stale executions from finalizing.
        Returns number of reaped jobs.
        """
        lease_dur = lease_duration_seconds if lease_duration_seconds is not None else self.lease_duration
        effective_max = max_attempts if max_attempts is not None else self.max_attempts

        cutoff = timezone.now() - timedelta(seconds=lease_dur)
        stale_jobs = GenerationJob.objects.filter(
            status=GenerationJob.Status.RUNNING,
            locked_at__lt=cutoff
        )

        reaped_count = 0
        for job in stale_jobs:
            job_max = min(job.max_attempts, effective_max)
            if job.attempts < job_max:
                logger.warning(
                    "[%s] Requeuing abandoned job %s (attempt %d/%d; lease expired at %s)",
                    self.worker_id, job.id, job.attempts, job_max, job.locked_at
                )
                job.status = GenerationJob.Status.PENDING
                job.locked_at = None
                job.locked_by = ""
                job.claim_token = None
                job.progress_message = "Previous worker timed out; job requeued for execution."
                job.save(update_fields=['status', 'locked_at', 'locked_by', 'claim_token', 'progress_message'])
            else:
                logger.error(
                    "[%s] Marking abandoned job %s as FAILED (exhausted %d max attempts)",
                    self.worker_id, job.id, job_max
                )
                job.status = GenerationJob.Status.FAILED
                job.locked_at = None
                job.locked_by = ""
                job.claim_token = None
                job.finished_at = timezone.now()
                job.error_message = "Worker execution lease expired after maximum retry attempts."
                job.progress_message = "Generation timed out."
                job.save(update_fields=['status', 'locked_at', 'locked_by', 'claim_token', 'finished_at', 'error_message', 'progress_message'])
            reaped_count += 1

        return reaped_count

    def claim_next_job(self, max_attempts: int = None) -> GenerationJob | None:
        """
        Claims the oldest PENDING job atomically using SELECT ... FOR UPDATE SKIP LOCKED.
        Generates a fresh UUID claim token for optimistic fencing.
        Immediately commits and releases the row lock, setting status to RUNNING.
        """
        with transaction.atomic():
            job = (
                GenerationJob.objects
                .select_for_update(skip_locked=True)
                .filter(status=GenerationJob.Status.PENDING)
                .order_by('created_at')
                .first()
            )
            if job:
                now = timezone.now()
                claim_token = uuid.uuid4()
                job.status = GenerationJob.Status.RUNNING
                job.claim_token = claim_token
                job.started_at = now
                job.locked_at = now
                job.locked_by = self.worker_id
                job.attempts += 1
                job.step = 1
                job.progress = 15
                job.progress_message = f"Initializing curriculum architect for '{job.subject_name}'..."
                job.save(update_fields=[
                    'status', 'claim_token', 'started_at', 'locked_at', 'locked_by',
                    'attempts', 'step', 'progress', 'progress_message'
                ])
                return job
        return None

    def execute_job(self, job: GenerationJob, heartbeat_interval: int = None, generator_func=None):
        """
        Executes AI curriculum generation for a claimed job outside row-level lock transactions.
        Uses optimistic fencing token checks before state transitions and activates JobHeartbeat
        to maintain lease liveness during long-running Gemini API calls.
        """
        job_id = job.id
        current_claim_token = job.claim_token
        hb_interval = heartbeat_interval if heartbeat_interval is not None else self.heartbeat_interval
        subject_name = (job.subject_name or "").strip()
        subject_code = (job.subject_code or "").strip()
        syllabus_text = (job.syllabus_text or "").strip()
        generate_fn = generator_func or generate_curriculum_from_prompt

        try:
            # Step 2: Syllabus Source Grounding & Prompt Synthesis with fencing verification
            updated = GenerationJob.objects.filter(
                id=job_id,
                claim_token=current_claim_token,
                status=GenerationJob.Status.RUNNING
            ).update(
                step=2,
                progress=40,
                progress_message="Parsing syllabus outline and preparing source-grounded model prompt...",
                locked_at=timezone.now()
            )
            if updated == 0:
                logger.warning("[%s] Job %s lost claim token before Step 2. Aborting execution.", self.worker_id, job_id)
                return

            # Step 3: Gemini Generation with fencing verification
            updated = GenerationJob.objects.filter(
                id=job_id,
                claim_token=current_claim_token,
                status=GenerationJob.Status.RUNNING
            ).update(
                step=3,
                progress=70,
                progress_message="Generating 5 structured modules and learning keywords with AI...",
                locked_at=timezone.now()
            )
            if updated == 0:
                logger.warning("[%s] Job %s lost claim token before Step 3. Aborting execution.", self.worker_id, job_id)
                return

            # Execute generation service protected by active heartbeat thread
            with JobHeartbeat(job_id, current_claim_token, hb_interval, self.worker_id) as heartbeat:
                try:
                    subject = generate_fn(
                        subject_name=subject_name,
                        subject_code=subject_code,
                        syllabus_text=syllabus_text,
                        job_id=job_id,
                        claim_token=current_claim_token,
                        lease_duration=self.lease_duration
                    )
                except TypeError:
                    subject = generate_fn(
                        subject_name=subject_name,
                        subject_code=subject_code,
                        syllabus_text=syllabus_text
                    )
                if heartbeat.lost_event.is_set():
                    logger.warning(
                        "[%s] Heartbeat was lost during generation for job %s (token %s revoked). Discarding output.",
                        self.worker_id, job_id, current_claim_token
                    )
                    return

            # Step 4: Finalizing & Completed with Atomic Fencing Check
            # When generate_fn is generate_curriculum_from_prompt, persist_curriculum_records
            # has already verified fencing and marked the job COMPLETED under row lock.
            # If generate_fn was a mock returning a Subject instance directly, handle completion atomically:
            job.refresh_from_db()
            if job.status == GenerationJob.Status.RUNNING:
                with transaction.atomic():
                    locked_job = GenerationJob.objects.select_for_update().filter(
                        id=job_id,
                        claim_token=current_claim_token,
                        status=GenerationJob.Status.RUNNING
                    ).first()
                    if locked_job and not locked_job.is_lease_expired(lease_duration_seconds=self.lease_duration):
                        locked_job.step = 4
                        locked_job.progress = 100
                        locked_job.status = GenerationJob.Status.COMPLETED
                        if isinstance(subject, Subject):
                            locked_job.subject = subject
                        locked_job.finished_at = timezone.now()
                        modules_count = subject.modules.count() if hasattr(subject, 'modules') else 0
                        subj_name = getattr(subject, 'name', locked_job.subject_name)
                        subj_code = getattr(subject, 'code', locked_job.subject_code)
                        locked_job.progress_message = (
                            f"Syllabus for '{subj_name}' ({subj_code}) generated successfully with "
                            f"{modules_count} modules!"
                        )
                        locked_job.error_message = ""
                        locked_job.save(update_fields=[
                            'step', 'progress', 'status', 'subject', 'finished_at', 'progress_message', 'error_message'
                        ])
                    else:
                        logger.warning(
                            "[%s] Stale execution detected for job %s. Claim token %s was revoked. Discarding result.",
                            self.worker_id, job_id, current_claim_token
                        )
                        return

            logger.info(
                "[%s] Job %s COMPLETED successfully. Subject '%s' (%s) -> /subjects/%s/",
                self.worker_id, job_id,
                getattr(subject, 'name', job.subject_name),
                getattr(subject, 'code', job.subject_code),
                getattr(subject, 'slug', '')
            )

        except CurriculumFencingError as fe:
            logger.warning(
                "[%s] Job %s fencing validation failed (revoked claim token or expired lease): %s. Discarding output.",
                self.worker_id, job_id, fe
            )
            return

        except GeminiTransientError as e:
            err_msg = "The AI curriculum service is temporarily experiencing high server demand. Please wait a few moments and try again."
            logger.warning("[%s] Job %s transient error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiAuthError as e:
            err_msg = "AI service authentication error. Please verify that your GROQ_API_KEY (or GEMINI_API_KEY) is configured correctly."
            logger.error("[%s] Job %s auth error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiInvalidRequestError as e:
            err_msg = f"Unable to process curriculum request: {e}"
            logger.error("[%s] Job %s invalid request: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiModelNotFoundError as e:
            err_msg = "The configured AI model is not available. Please verify your AI model configuration."
            logger.error("[%s] Job %s model not found: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except CurriculumValidationError as ve:
            err_msg = f"Syllabus validation failed: {ve} Please verify that your syllabus contains 5 clear module headings."
            logger.warning("[%s] Job %s validation error: %s", self.worker_id, job_id, ve)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except CurriculumParseError as pe:
            err_msg = "The AI generated an incomplete curriculum structure. Please try again or provide detailed syllabus text."
            logger.warning("[%s] Job %s parse error: %s", self.worker_id, job_id, pe)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except CurriculumGenerationError as ge:
            err_msg = str(ge)
            logger.error("[%s] Job %s generation error: %s", self.worker_id, job_id, ge)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except Exception as exc:
            err_msg = "An unexpected error occurred while generating the syllabus. Please try again."
            logger.exception("[%s] Job %s unexpected error: %s", self.worker_id, job_id, exc)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

    def _mark_job_failed(self, job_id, current_claim_token, err_msg: str):
        """Marks a job as FAILED only if current_claim_token still holds ownership."""
        try:
            rows_updated = GenerationJob.objects.filter(
                id=job_id,
                claim_token=current_claim_token,
                status=GenerationJob.Status.RUNNING
            ).update(
                status=GenerationJob.Status.FAILED,
                finished_at=timezone.now(),
                error_message=err_msg,
                progress_message="Curriculum generation encountered an error."
            )
            if rows_updated == 0:
                logger.warning(
                    "[%s] Stale failure recording ignored for job %s. Claim token %s is no longer valid.",
                    self.worker_id, job_id, current_claim_token
                )
        except Exception as e:
            logger.exception("[%s] Failed to record failure state for job %s: %s", self.worker_id, job_id, e)


class InProcessWorkerThread(threading.Thread):
    """
    Managed background worker thread running inside Gunicorn web worker processes.
    Picks up jobs immediately on notification event, reaps leases, and falls back to polling.
    """
    def __init__(self, engine: GenerationWorkerEngine, shutdown_event: threading.Event, job_event: threading.Event):
        super().__init__(name=f"in-process-worker-{engine.worker_id}", daemon=True)
        self.engine = engine
        self.shutdown_event = shutdown_event
        self.job_event = job_event

    def run(self):
        logger.info(
            "[%s] In-process curriculum generation worker started. poll_interval=%.1fs, lease_duration=%ds, heartbeat_interval=%ds",
            self.engine.worker_id, self.engine.poll_interval, self.engine.lease_duration, self.engine.heartbeat_interval
        )
        try:
            while not self.shutdown_event.is_set():
                try:
                    self.engine.close_db_connections()

                    # 1. Recover abandoned jobs where lease expired
                    self.engine.reap_abandoned_jobs()

                    # 2. Process all pending jobs sequentially
                    job_processed = True
                    while job_processed and not self.shutdown_event.is_set():
                        job = self.engine.claim_next_job()
                        if job:
                            logger.info(
                                "[%s] Claimed job %s (token %s) for '%s'",
                                self.engine.worker_id, job.id, job.claim_token, job.subject_name
                            )
                            self.engine.execute_job(job)
                            self.engine.close_db_connections()
                        else:
                            job_processed = False

                    # 3. Wait on job_event or poll_interval
                    if not self.shutdown_event.is_set():
                        self.engine.close_db_connections()
                        woken = self.job_event.wait(timeout=self.engine.poll_interval)
                        if woken:
                            self.job_event.clear()

                except DatabaseError as dbe:
                    logger.error("[%s] Database error in worker thread: %s", self.engine.worker_id, dbe)
                    self.engine.close_db_connections()
                    self.shutdown_event.wait(timeout=min(self.engine.poll_interval, 5.0))
                except Exception as exc:
                    logger.exception("[%s] Unexpected error in worker thread: %s", self.engine.worker_id, exc)
                    self.engine.close_db_connections()
                    self.shutdown_event.wait(timeout=min(self.engine.poll_interval, 5.0))
        finally:
            self.engine.close_db_connections()
            try:
                from django.db import connection
                connection.close()
            except Exception:
                pass
            logger.info("[%s] In-process curriculum generation worker thread exited cleanly.", self.engine.worker_id)


def start_in_process_worker(engine: GenerationWorkerEngine = None) -> InProcessWorkerThread:
    """
    Starts the in-process background worker thread if not already running.
    Thread-safe and idempotent within a process: calling it multiple times will not duplicate threads.
    """
    global _worker_thread, _shutdown_event, _job_posted_event
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            logger.debug("In-process generation worker thread is already running.")
            return _worker_thread

        _shutdown_event.clear()
        _job_posted_event.clear()
        worker_engine = engine or GenerationWorkerEngine()
        _worker_thread = InProcessWorkerThread(
            engine=worker_engine,
            shutdown_event=_shutdown_event,
            job_event=_job_posted_event
        )
        _worker_thread.start()
        logger.info("Successfully started in-process generation worker thread [%s].", _worker_thread.name)
        return _worker_thread


def stop_in_process_worker(timeout: float = 15.0) -> bool:
    """
    Signals the in-process background worker thread to stop and waits up to timeout seconds.
    Returns True if thread terminated cleanly, False if it timed out.
    """
    global _worker_thread, _shutdown_event, _job_posted_event
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            logger.debug("In-process generation worker thread is not running.")
            return True

        logger.info("Signaling in-process worker thread [%s] to stop (timeout=%.1fs)...", _worker_thread.name, timeout)
        _shutdown_event.set()
        _job_posted_event.set()  # Wake thread immediately so it can observe shutdown_event
        _worker_thread.join(timeout=timeout)

        is_stopped = not _worker_thread.is_alive()
        if is_stopped:
            logger.info("In-process worker thread [%s] stopped cleanly.", _worker_thread.name)
            _worker_thread = None
        else:
            logger.warning(
                "In-process worker thread [%s] did not exit within %.1fs timeout. It will be abandoned on process termination.",
                _worker_thread.name, timeout
            )
        return is_stopped


def notify_new_job():
    """
    Notifies the in-process worker thread that a new PENDING job has been enqueued.
    Wakes the thread immediately (<1ms) instead of waiting for the polling interval.
    Safe to call even if no worker is running.
    """
    _job_posted_event.set()


def get_worker_thread() -> InProcessWorkerThread | None:
    """Returns the active worker thread, or None if not running."""
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        return None


def is_worker_accepting_jobs() -> bool:
    """
    Checks if the in-process background worker is active and healthy to accept new jobs.
    Performs thread health check and executes safe supervisor revival if the thread
    died unexpectedly.
    Returns True if worker is healthy and ready; False otherwise.
    """
    if getattr(settings, 'SIMULATE_WORKER_OFFLINE', False):
        return False

    with _worker_lock:
        if _shutdown_event.is_set():
            return False

        # If thread exists and is alive, worker is healthy
        if _worker_thread is not None and _worker_thread.is_alive():
            return True

        # If running in Django test suite without explicit start, allow test bypass
        if ('test' in sys.argv or getattr(settings, 'TESTING', False)) and _worker_thread is None:
            return True

        # Thread is dead or was not started; attempt supervisor revival
        logger.warning(
            "In-process worker thread is not running or has died unexpectedly. Attempting supervisor revival..."
        )
        try:
            thread = start_in_process_worker()
            return thread is not None and thread.is_alive()
        except Exception as e:
            logger.exception("Supervisor failed to revive in-process worker thread: %s", e)
            return False
