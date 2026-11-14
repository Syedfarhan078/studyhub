import os
import sys
import time
import uuid
import signal
import logging
from datetime import timedelta

from django.core.management.base import BaseCommand
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


class Command(BaseCommand):
    help = "Runs the standalone PostgreSQL-backed AI curriculum generation worker."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_requested = False
        self.worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    def add_arguments(self, parser):
        parser.add_argument(
            '--poll-interval',
            type=float,
            default=2.0,
            help="Interval in seconds to poll for pending jobs when idle (default: 2.0s)."
        )
        parser.add_argument(
            '--lease-duration',
            type=int,
            default=300,
            help="Duration in seconds after which an unrenewed running job is deemed abandoned (default: 300s)."
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
        max_jobs = options['max_jobs']
        run_once = options['once']

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "[%s] StudyHub Generation Worker started. poll_interval=%.1fs, lease_duration=%ds, max_jobs=%s, once=%s",
            self.worker_id, poll_interval, lease_duration, max_jobs, run_once
        )
        self.stdout.write(self.style.SUCCESS(
            f"[{self.worker_id}] StudyHub Generation Worker started successfully. Waiting for jobs..."
        ))

        jobs_processed = 0

        while not self.shutdown_requested:
            try:
                # 1. Recover abandoned jobs where lease expired
                self.reap_abandoned_jobs(lease_duration)

                # 2. Claim next available job atomically
                job = self.claim_next_job()

                if job:
                    logger.info("[%s] Claimed job %s for subject '%s'", self.worker_id, job.id, job.subject_name)
                    self.execute_job(job)
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
        """Close old database connections to prevent pool exhaustion or stale sockets."""
        if 'test' in sys.argv or getattr(settings, 'TESTING', False):
            return
        try:
            close_old_connections()
        except Exception as e:
            logger.debug("[%s] Error closing db connections: %s", self.worker_id, e)

    def reap_abandoned_jobs(self, lease_duration_seconds: int):
        """
        Detects jobs stuck in RUNNING status whose worker lease has expired.
        Requeues them if retries remain; marks them FAILED if attempts exhausted.
        """
        cutoff = timezone.now() - timedelta(seconds=lease_duration_seconds)
        stale_jobs = GenerationJob.objects.filter(
            status=GenerationJob.Status.RUNNING,
            locked_at__lt=cutoff
        )

        for job in stale_jobs:
            if job.attempts < job.max_attempts:
                logger.warning(
                    "[%s] Requeuing abandoned job %s (attempt %d/%d; lease expired at %s)",
                    self.worker_id, job.id, job.attempts, job.max_attempts, job.locked_at
                )
                job.status = GenerationJob.Status.PENDING
                job.locked_at = None
                job.locked_by = ""
                job.progress_message = "Previous worker timed out; job requeued for execution."
                job.save(update_fields=['status', 'locked_at', 'locked_by', 'progress_message'])
            else:
                logger.error(
                    "[%s] Marking abandoned job %s as FAILED (exhausted %d max attempts)",
                    self.worker_id, job.id, job.max_attempts
                )
                job.status = GenerationJob.Status.FAILED
                job.finished_at = timezone.now()
                job.error_message = "Worker execution lease expired after maximum retry attempts."
                job.progress_message = "Generation timed out."
                job.save(update_fields=['status', 'finished_at', 'error_message', 'progress_message'])

    def claim_next_job(self):
        """
        Claims the oldest PENDING job atomically using SELECT ... FOR UPDATE SKIP LOCKED.
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
                job.status = GenerationJob.Status.RUNNING
                job.started_at = now
                job.locked_at = now
                job.locked_by = self.worker_id
                job.attempts += 1
                job.step = 1
                job.progress = 15
                job.progress_message = f"Initializing curriculum architect for '{job.subject_name}'..."
                job.save(update_fields=[
                    'status', 'started_at', 'locked_at', 'locked_by',
                    'attempts', 'step', 'progress', 'progress_message'
                ])
                return job
        return None

    def execute_job(self, job: GenerationJob):
        """
        Executes AI curriculum generation for a claimed job outside of row-level lock transactions.
        Updates progress states and handles failures gracefully without unhandled exceptions.
        """
        job_id = job.id
        subject_name = (job.subject_name or "").strip()
        subject_code = (job.subject_code or "").strip()
        syllabus_text = (job.syllabus_text or "").strip()

        try:
            # Step 2: Syllabus Source Grounding & Prompt Synthesis
            job.step = 2
            job.progress = 40
            job.progress_message = "Parsing syllabus outline and preparing source-grounded model prompt..."
            job.locked_at = timezone.now()
            job.save(update_fields=['step', 'progress', 'progress_message', 'locked_at'])

            # Step 3: Gemini Generation
            job.step = 3
            job.progress = 70
            job.progress_message = "Generating 5 structured modules and learning keywords with Gemini AI..."
            job.locked_at = timezone.now()
            job.save(update_fields=['step', 'progress', 'progress_message', 'locked_at'])

            # Execute generation service (strictly source-grounded and validated)
            subject = generate_curriculum_from_prompt(
                subject_name=subject_name,
                subject_code=subject_code,
                syllabus_text=syllabus_text
            )

            # Step 4: Finalizing & Completed
            job.step = 4
            job.progress = 100
            job.status = GenerationJob.Status.COMPLETED
            job.subject = subject
            job.finished_at = timezone.now()
            job.progress_message = (
                f"Syllabus for '{subject.name}' ({subject.code}) generated successfully with "
                f"{subject.modules.count()} modules!"
            )
            job.error_message = ""
            job.save(update_fields=[
                'step', 'progress', 'status', 'subject', 'finished_at',
                'progress_message', 'error_message'
            ])
            logger.info(
                "[%s] Job %s COMPLETED successfully. Subject '%s' (%s) -> /subjects/%s/",
                self.worker_id, job_id, subject.name, subject.code, subject.slug
            )

        except GeminiTransientError as e:
            err_msg = "The AI curriculum service is temporarily experiencing high server demand. Please wait a few moments and try again."
            logger.warning("[%s] Job %s transient error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job, err_msg)

        except GeminiAuthError as e:
            err_msg = "AI service authentication error. Please verify that your GEMINI_API_KEY is configured correctly."
            logger.error("[%s] Job %s auth error: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job, err_msg)

        except GeminiInvalidRequestError as e:
            err_msg = f"Unable to process curriculum request: {e}"
            logger.error("[%s] Job %s invalid request: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job, err_msg)

        except GeminiModelNotFoundError as e:
            err_msg = "The configured AI model is not available. Please verify your Gemini model configuration."
            logger.error("[%s] Job %s model not found: %s", self.worker_id, job_id, e)
            self._mark_job_failed(job, err_msg)

        except CurriculumValidationError as ve:
            err_msg = f"Syllabus validation failed: {ve} Please verify that your syllabus contains 5 clear module headings."
            logger.warning("[%s] Job %s validation error: %s", self.worker_id, job_id, ve)
            self._mark_job_failed(job, err_msg)

        except CurriculumParseError as pe:
            err_msg = "The AI generated an incomplete curriculum structure. Please try again or provide detailed syllabus text."
            logger.warning("[%s] Job %s parse error: %s", self.worker_id, job_id, pe)
            self._mark_job_failed(job, err_msg)

        except CurriculumGenerationError as ge:
            err_msg = str(ge)
            logger.error("[%s] Job %s generation error: %s", self.worker_id, job_id, ge)
            self._mark_job_failed(job, err_msg)

        except Exception as exc:
            err_msg = "An unexpected error occurred while generating the syllabus. Please try again."
            logger.exception("[%s] Job %s unexpected error: %s", self.worker_id, job_id, exc)
            self._mark_job_failed(job, err_msg)

    def _mark_job_failed(self, job: GenerationJob, err_msg: str):
        """Marks a job as FAILED and persists sanitized error details."""
        try:
            job.status = GenerationJob.Status.FAILED
            job.finished_at = timezone.now()
            job.error_message = err_msg
            job.progress_message = "Curriculum generation encountered an error."
            job.save(update_fields=['status', 'finished_at', 'error_message', 'progress_message'])
        except Exception as e:
            logger.exception("[%s] Failed to record failure state for job %s: %s", self.worker_id, job.id, e)
