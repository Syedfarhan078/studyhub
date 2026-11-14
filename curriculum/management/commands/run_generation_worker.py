import os
import sys
import time
import signal
import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError
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
from curriculum.worker import (
    JobHeartbeat,
    GenerationWorkerEngine,
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Runs the standalone PostgreSQL-backed AI curriculum generation worker."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_requested = False
        self.engine = GenerationWorkerEngine()

    @property
    def worker_id(self):
        return self.engine.worker_id

    @worker_id.setter
    def worker_id(self, value):
        self.engine.worker_id = value

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

    def close_db_connections(self):
        self.engine.close_db_connections()

    def reap_abandoned_jobs(self, lease_duration_seconds: int = None, max_attempts: int = 3):
        return self.engine.reap_abandoned_jobs(lease_duration_seconds, max_attempts)

    def claim_next_job(self, max_attempts: int = 3):
        return self.engine.claim_next_job(max_attempts)

    def execute_job(self, job: GenerationJob, heartbeat_interval: int = 30):
        return self.engine.execute_job(
            job=job,
            heartbeat_interval=heartbeat_interval,
            generator_func=generate_curriculum_from_prompt
        )

    def _mark_job_failed(self, job_id, current_claim_token, err_msg: str):
        return self.engine._mark_job_failed(job_id, current_claim_token, err_msg)

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

        # Configure engine parameters
        self.engine.poll_interval = poll_interval
        self.engine.lease_duration = lease_duration
        self.engine.heartbeat_interval = heartbeat_interval
        self.engine.max_attempts = max_attempts

        # Register signal handlers for graceful shutdown on CLI process
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
