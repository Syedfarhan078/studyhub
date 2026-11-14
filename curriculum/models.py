from django.db import models
from django.conf import settings
from django.utils import timezone
import os
import uuid

# Create your models here.

class Subject(models.Model):
    name = models.CharField(max_length=150, help_text="Full subject name, e.g. Machine Learning")
    code = models.CharField(max_length=20, help_text="Subject code or abbreviation, e,g. ML, DL, BAI701")
    slug = models.SlugField(max_length=160, unique=True, help_text="URL-Freindly identifier, e.g. machine-learning")
    description = models.TextField(blank=True, help_text="Sjort overview of what this subject covers.")
    order = models.PositiveSmallIntegerField(default=1, help_text="Display order on the homepage")

    class Meta:
        ordering = ['order', 'name']
        verbose_name = 'Subject'
        verbose_name_plural = "Subjects"

    def __str__(self):
        return f"{self.code} - {self.name}"


class Module(models.Model):
    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        related_name='modules',
        help_text='The subject this module belongs to'
    )
    module_number = models.PositiveSmallIntegerField(help_text="Module sequence number (e.g. 1, 2, 3, 4, 5)")

    title = models.CharField(
        max_length=100,
        help_text="Modeul title, e.g. Introduction to ML"
    )

    description = models.TextField(
        blank=True,
        help_text="Breif summary of the module's scopes"
    )

    class Meta:
        ordering = ['subject', 'module_number']
        unique_together = ['subject', 'module_number']
        verbose_name = "Module"
        verbose_name_plural = "Modules"

    def __str__(self):
        return f"{self.subject.code} - Module {self.module_number}: {self.title}"


class Topic(models.Model):
    module = models.ForeignKey(
        Module,
        on_delete=models.CASCADE,
        related_name="topics",
        help_text="The module this topic belongs to"
    )

    title = models.CharField(
        max_length=200,
        help_text="Topic title, e.g. Supervised Learning"
    )

    slug = models.SlugField(
        max_length=220,
        help_text="URL-Friendly Indentifier"
    )

    order = models.PositiveSmallIntegerField(
        default = 1,
        help_text = "Display order within the module"
    )

    description = models.TextField(
        blank=True,
        help_text="Breif notes or syllabus details for this topic"
    )

    class Meta:
        ordering = ['module', 'order', 'title']
        unique_together = ['module', 'slug']
        verbose_name = "Topic"
        verbose_name_plural = "Topics"

    def __str__(self):
        return f"{self.module.subject.code} - M{self.module.module_number}: {self.title}"


class TopicKeyword(models.Model):
    topic = models.ForeignKey(
        Topic,
        on_delete=models.CASCADE,
        related_name="keywords",
        help_text="The topic this keyword helps discover resources for"
    )    

    keyword = models.CharField(
        max_length=120,
        help_text="Search keyword or phrase, e.g. 'supervised learning maching learning tutorial'"
    )

    class Meta:
        ordering = ['keyword']
        unique_together = ['topic', 'keyword']
        verbose_name = "Topic Keyword"
        verbose_name_plural = "Topic Keywords"

    def __str__(self):
        return f"{self.topic.title} -> {self.keyword}"


class ModuleNote(models.Model):
    module = models.ForeignKey(
        Module,
        on_delete=models.CASCADE,
        related_name='notes',
        help_text="The syllabus module this note belongs to"
    )
    title = models.CharField(
        max_length=200,
        help_text="Title or description of the note (e.g. VTU Handwritten Notes - Module 1)"
    )
    file = models.FileField(
        upload_to='module_notes/%Y/%m/',
        help_text="Upload PDF, DOCX, or PPT notes"
    )
    description = models.TextField(
        blank=True,
        help_text="Optional description, author, or exam tips"
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp of when the note was uploaded"
    )

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Module Note"
        verbose_name_plural = "Module Notes"

    def __str__(self):
        try:
            return f"{self.module.subject.code} - M{self.module.module_number}: {self.title}"
        except Exception:
            return self.title or "Module Note"

    @property
    def file_extension(self):
        """Returns the file extension in uppercase (e.g. 'PDF', 'DOCX')."""
        try:
            if not self.file or not self.file.name:
                return "PDF"
            name, extension = os.path.splitext(self.file.name)
            ext = extension.replace('.', '').upper()
            return ext if ext else "PDF"
        except Exception:
            return "PDF"

    @property
    def file_size_display(self):
        """Calculates and returns human-readable file size safely without raising TypeError."""
        try:
            if not self.file:
                return "Unknown"
            bytes_size = self.file.size
            if bytes_size is None or bytes_size <= 0:
                return "Unknown"
            if bytes_size < 1024:
                return f"{bytes_size} B"
            elif bytes_size < 1024 * 1024:
                return f"{bytes_size / 1024:.1f} KB"
            else:
                return f"{bytes_size / (1024 * 1024):.1f} MB"
        except Exception:
            return "Unknown"


class TopicProgress(models.Model):
    class Status(models.TextChoices):
        NOT_STARTED = 'NOT_STARTED', 'Not Started'
        IN_PROGRESS = 'IN_PROGRESS', 'In Progress'
        COMPLETED = 'COMPLETED', 'Completed'

    topic = models.OneToOneField(
        Topic,
        on_delete=models.CASCADE,
        related_name='progress',
        help_text="The syllabus topic being tracked"
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.NOT_STARTED,
        help_text="Current learning progress"
    )
    personal_notes = models.TextField(
        blank=True,
        help_text="Personal study notes, formula reminders, and exam tips"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last time progress or notes were updated"
    )

    class Meta:
        verbose_name = "Topic Progress"
        verbose_name_plural = "Topic Progress Records"

    def __str__(self):
        try:
            return f"{self.topic.title} - {self.get_status_display()}"
        except Exception:
            return f"Topic Progress #{self.pk}"


class GenerationJob(models.Model):
    class Status(models.TextChoices):
        PENDING = 'PENDING', 'Pending'
        RUNNING = 'RUNNING', 'Running'
        COMPLETED = 'COMPLETED', 'Completed'
        FAILED = 'FAILED', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='generation_jobs',
        help_text="User who initiated this job (optional for anonymous/guest generation)"
    )
    subject = models.ForeignKey(
        Subject,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='generation_jobs',
        help_text="Subject record created once curriculum generation succeeds"
    )
    subject_name = models.CharField(max_length=150, help_text="Full subject name submitted")
    subject_code = models.CharField(max_length=20, blank=True, help_text="Optional subject code")
    syllabus_text = models.TextField(blank=True, help_text="Prescribed syllabus text if provided")

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
        help_text="Current processing status"
    )
    progress = models.PositiveSmallIntegerField(default=0, help_text="Progress percentage (0-100)")
    step = models.PositiveSmallIntegerField(default=1, help_text="Current step number")
    total_steps = models.PositiveSmallIntegerField(default=4, help_text="Total number of steps")
    progress_message = models.CharField(
        max_length=255,
        default="Job created and pending worker pickup.",
        help_text="User-facing status message"
    )
    attempts = models.PositiveSmallIntegerField(default=0, help_text="Number of execution attempts")
    max_attempts = models.PositiveSmallIntegerField(default=3, help_text="Maximum allowed execution attempts")
    error_message = models.TextField(blank=True, help_text="Sanitized error description on failure")

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    locked_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Timestamp when worker claimed/heartbeat the job"
    )
    locked_by = models.CharField(
        max_length=100,
        blank=True,
        help_text="Worker identifier currently executing this job"
    )
    claim_token = models.UUIDField(
        null=True,
        blank=True,
        editable=False,
        db_index=True,
        help_text="Unique fencing token for active execution attempt"
    )
    session_key = models.CharField(
        max_length=40,
        blank=True,
        db_index=True,
        help_text="Session key of anonymous creator for secure ownership binding"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['locked_at']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['claim_token']),
            models.Index(fields=['session_key']),
        ]
        verbose_name = "Generation Job"
        verbose_name_plural = "Generation Jobs"

    def __str__(self):
        return f"GenerationJob {self.id} [{self.status}] - {self.subject_name}"

    def is_lease_expired(self, lease_duration_seconds=300):
        if not self.locked_at:
            return True
        return (timezone.now() - self.locked_at).total_seconds() > lease_duration_seconds

