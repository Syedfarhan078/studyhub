from django.db import models
import os

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
        return f"{self.topic.title} - {self.get_status_display()}"
