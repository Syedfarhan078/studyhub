import logging
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.conf import settings
from django.db.models import Q

from curriculum.models import GenerationJob

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Safely deletes old COMPLETED and FAILED GenerationJob audit records beyond retention cutoff."

    def add_arguments(self, parser):
        default_days = getattr(settings, 'GENERATION_JOB_RETENTION_DAYS', 30)
        parser.add_argument(
            '--days',
            type=int,
            default=default_days,
            help=f"Number of days to retain finished generation jobs (default: {default_days} days)."
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            default=False,
            help="Simulate cleanup without deleting any records from the database."
        )

    def handle(self, *args, **options):
        days = options['days']
        dry_run = options['dry_run']

        if days <= 0:
            raise CommandError("--days must be a positive integer greater than 0.")

        cutoff = timezone.now() - timedelta(days=days)
        logger.info(
            "Starting GenerationJob cleanup (cutoff: %s, days: %d, dry_run: %s)",
            cutoff, days, dry_run
        )

        # Strictly target only COMPLETED or FAILED jobs older than cutoff.
        # NEVER match PENDING or RUNNING jobs regardless of timestamp.
        eligible_qs = GenerationJob.objects.filter(
            status__in=[GenerationJob.Status.COMPLETED, GenerationJob.Status.FAILED]
        ).filter(
            Q(finished_at__lt=cutoff) | Q(finished_at__isnull=True, created_at__lt=cutoff)
        )

        count = eligible_qs.count()

        if dry_run:
            msg = (
                f"[DRY-RUN] Found {count} finished generation jobs older than {days} days "
                f"(cutoff: {cutoff.strftime('%Y-%m-%d %H:%M:%S UTC')}) that would be deleted. "
                f"No records were modified."
            )
            logger.info(msg)
            self.stdout.write(self.style.WARNING(msg))
            return

        # Execute safe bulk delete on eligible queue jobs only.
        # Associated Subject, Module, Topic, and user records are completely preserved.
        deleted_count, details = eligible_qs.delete()
        jobs_deleted = details.get('curriculum.GenerationJob', deleted_count)

        msg = (
            f"Successfully deleted {jobs_deleted} finished generation jobs older than {days} days "
            f"(cutoff: {cutoff.strftime('%Y-%m-%d %H:%M:%S UTC')})."
        )
        logger.info(msg)
        self.stdout.write(self.style.SUCCESS(msg))
