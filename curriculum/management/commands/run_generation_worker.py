import os
import sys
import time
import uuid
import signal
import logging
import threading
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction, close_old_connections, DatabaseError
from django.utils import timezone
from django.conf import settings

from curriculum.models import GenerationJob, Subject
from curriculum.services.ai_curriculum import (
    generate_curriculum_from_prompt,
    CurriculumGenerationError,
    GeminiTransientError,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    CurriculumParseError,
    CurriculumValidationError,
)

logger = logging.getLogger(__name__)


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

    def _close_conns(self):
        if 'test' in sys.argv or getattr(settings, 'TESTING', False):
            return
        try:
            close_old_connections()
        except Exception:
            pass

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                try:
                    self._close_conns()
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
                    self._close_conns()
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
        self._close_conns()


class Command(BaseCommand):
    help = "Runs the standalone PostgreSQL-backed AI curriculum generation worker."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_requested = False
        self.worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    def add_arguments(self, parser):
        default_poll = getattr(settings, 'GENERATION_WORKER_POLL_INTERVAL', 2.0)
        default_lease = getattr(settings, 'GENERATION_WORKER_LEASE_DURATION', 300)
        default_heartbeat = getattr(settings, 'GENERATION_WORKER_HEARTBEAT_INTERVAL', 30)
        default_max_attempts = getattr(settings, 'GENERATION_WORKER_MAX_ATTEMPTS', 3)

        parser.add_argument(
            '--poll-interval',
            type=float,
            default=default_poll,
            help=f"Interval in seconds to poll for pending jobs when idle (default: {default_poll}s)."
        )
        parser.add_argument(
            '--lease-duration',
            type=int,
            default=default_lease,
            help=f"Duration in seconds after which an unrenewed running job is deemed abandoned (default: {default_lease}s)."
        )
        parser.add_argument(
            '--heartbeat-interval',
            type=int,
            default=default_heartbeat,
            help=f"Interval in seconds between lease heartbeat refreshes during execution (default: {default_heartbeat}s)."
        )
        parser.add_argument(
            '--max-attempts',
            type=int,
            default=default_max_attempts,
            help=f"Maximum allowed execution attempts before marking an abandoned job FAILED (default: {default_max_attempts})."
        )
        parser.add_argument(
            '--max-jobs',
            type=int,
            default=None,
            help="Maximum number of jobs to process before exiting (default: None, runs continuously)."
        )
        parser.add_argument(
            '--once',
            action='store_true',
            default=False,
            help="Process at most one job if available, then exit."
        )

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info("[%s] Signal %s received. Graceful shutdown requested.", self.worker_id, sig_name)
        self.shutdown_requested = True

    def handle(self, *args, **options):
        poll_interval = options['poll_interval']
        lease_duration = options['lease_duration']
        heartbeat_interval = options['heartbeat_interval']
        max_attempts = options['max_attempts']
        max_jobs = options['max_jobs']
        run_once = options['once']

        # Validate configuration values
        if poll_interval <= 0:
            raise CommandError("--poll-interval must be greater than 0.")
        if lease_duration <= 0:
            raise CommandError("--lease-duration must be greater than 0.")
        if heartbeat_interval <= 0:
            raise CommandError("--heartbeat-interval must be greater than 0.")
        if lease_duration < heartbeat_interval * 2:
            raise CommandError(
                f"Invalid configuration: --lease-duration ({lease_duration}s) must be at least twice "
                f"the --heartbeat-interval ({heartbeat_interval}s) to prevent false-positive reclaiming."
            )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "[%s] StudyHub Generation Worker started. poll_interval=%.1fs, lease_duration=%ds, heartbeat_interval=%ds, max_jobs=%s, once=%s",
            self.worker_id, poll_interval, lease_duration, heartbeat_interval, max_jobs, run_once
        )
        self.stdout.write(self.style.SUCCESS(
            f"[{self.worker_id}] StudyHub Generation Worker started successfully. Waiting for jobs..."
        ))

        jobs_processed = 0

        while not self.shutdown_requested:
            try:
                # 1. Recover abandoned jobs where lease expired
                self.reap_abandoned_jobs(lease_duration, max_attempts)

                # 2. Claim next available job atomically
                job = self.claim_next_job(max_attempts)

                if job:
                    logger.info("[%s] Claimed job %s (token %s) for subject '%s'", self.worker_id, job.id, job.claim_token, job.subject_name)
                    self.execute_job(job, heartbeat_interval)
                    jobs_processed += 1

                    if run_once or (max_jobs is not None and jobs_processed >= max_jobs):
                        logger.info(
                            "[%s] Reached job target (%d jobs processed). Exiting worker.",
                            self.worker_id, jobs_processed
                        )
                        break
                else:
                    if run_once:
                        logger.info("[%s] Run-once requested and no pending jobs found. Exiting.", self.worker_id)
                        break

                    # Idle sleep with responsive shutdown check
                    self.close_db_connections()
                    for _ in range(int(poll_interval * 10)):
                        if self.shutdown_requested:
                            break
                        time.sleep(0.1)

            except DatabaseError as dbe:
                logger.error("[%s] Database error in worker main loop: %s", self.worker_id, dbe)
                self.close_db_connections()
                time.sleep(min(poll_interval, 5.0))
            except Exception as e:
                logger.exception("[%s] Unexpected exception in worker main loop: %s", self.worker_id, e)
                self.close_db_connections()
                time.sleep(min(poll_interval, 5.0))

        self.close_db_connections()
        logger.info("[%s] StudyHub Generation Worker exited cleanly. Total jobs processed: %d", self.worker_id, jobs_processed)
        self.stdout.write(self.style.SUCCESS(f"[{self.worker_id}] Worker stopped cleanly."))

    def close_db_connections(self):
        """Close old database connections in worker daemon mode to prevent stale sockets."""
        if 'test' in sys.argv or getattr(settings, 'TESTING', False):
            return
        try:
            close_old_connections()
        except Exception as e:
            logger.debug("[%s] Error closing db connections: %s", self.worker_id, e)

    def reap_abandoned_jobs(self, lease_duration_seconds: int, max_attempts: int = 3):
        """
        Detects jobs stuck in RUNNING status whose worker lease has expired.
        Requeues them if retries remain; marks them FAILED if attempts exhausted.
        Clears stale claim tokens to prevent stale executions from finalizing.
        """
        cutoff = timezone.now() - timedelta(seconds=lease_duration_seconds)
        stale_jobs = GenerationJob.objects.filter(
            status=GenerationJob.Status.RUNNING,
            locked_at__lt=cutoff
        )

        for job in stale_jobs:
            effective_max = min(job.max_attempts, max_attempts)
            if job.attempts < effective_max:
                logger.warning(
                    "[%s] Requeuing abandoned job %s (attempt %d/%d; lease expired at %s)",
                    self.worker_id, job.id, job.attempts, effective_max, job.locked_at
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
                    self.worker_id, job.id, effective_max
                )
                job.status = GenerationJob.Status.FAILED
                job.locked_at = None
                job.locked_by = ""
                job.claim_token = None
                job.finished_at = timezone.now()
                job.error_message = "Worker execution lease expired after maximum retry attempts."
                job.progress_message = "Generation timed out."
                job.save(update_fields=['status', 'locked_at', 'locked_by', 'claim_token', 'finished_at', 'error_message', 'progress_message'])

    def claim_next_job(self, max_attempts: int = 3):
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

    def execute_job(self, job: GenerationJob, heartbeat_interval: int = 30):
        """
        Executes AI curriculum generation for a claimed job outside of row-level lock transactions.
        Uses optimistic fencing token checks before state transitions and activates JobHeartbeat
        to maintain lease liveness during long-running Gemini API calls.
        """
        job_id = job.id
        current_claim_token = job.claim_token
        subject_name = (job.subject_name or "").strip()
        subject_code = (job.subject_code or "").strip()
        syllabus_text = (job.syllabus_text or "").strip()

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
                progress_message="Generating 5 structured modules and learning keywords with Gemini AI...",
                locked_at=timezone.now()
            )
            if updated == 0:
                logger.warning("[%s] Job %s lost claim token before Step 3. Aborting execution.", self.worker_id, job_id)
                return

            # Execute generation service protected by active heartbeat thread
            with JobHeartbeat(job_id, current_claim_token, heartbeat_interval, self.worker_id) as heartbeat:
                subject = generate_curriculum_from_prompt(
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
            with transaction.atomic():
                rows_updated = GenerationJob.objects.filter(
                    id=job_id,
                    claim_token=current_claim_token,
                    status=GenerationJob.Status.RUNNING
                ).update(
                    step=4,
                    progress=100,
                    status=GenerationJob.Status.COMPLETED,
                    subject=subject,
                    finished_at=timezone.now(),
                    progress_message=(
                        f"Syllabus for '{subject.name}' ({subject.code}) generated successfully with "
                        f"{subject.modules.count()} modules!"
                    ),
                    error_message=""
                )

            if rows_updated == 0:
                logger.warning(
                    "[%s] Stale execution detected for job %s. Claim token %s was revoked. Discarding result.",
                    self.worker_id, job_id, current_claim_token
                )
                return

            logger.info(
                "[%s] Job %s COMPLETED successfully. Subject '%s' (%s) -> /subjects/%s/",
                self.worker_id, job_id, subject.name, subject.code, subject.slug
            )

        except GeminiTransientError as e:
            err_msg = "The AI curriculum service is temporarily experiencing high server demand. Please wait a few moments and try again."
            logger.warning("[%s] Job %s transient error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiAuthError as e:
            err_msg = "AI service authentication error. Please verify that your GEMINI_API_KEY is configured correctly."
            logger.error("[%s] Job %s auth error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiInvalidRequestError as e:
            err_msg = f"Unable to process curriculum request: {e}"
            logger.error("[%s] Job %s invalid request: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job_id, current_claim_token, err_msg)

        except GeminiModelNotFoundError as e:
            err_msg = "The configured AI model is not available. Please verify your Gemini model configuration."
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
