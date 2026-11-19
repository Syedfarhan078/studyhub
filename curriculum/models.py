from django.db import models
from django.conf import settings
from django.utils import timezone
import os
import uuid

# Create your models here.

class Subject(models.Model):
    class Branch(models.TextChoices):
        ALL = 'ALL', 'All Branches / Common'
        CSE = 'CSE', 'Computer Science & Engineering'
        AIML = 'AIML', 'Artificial Intelligence & Machine Learning'
        ISE = 'ISE', 'Information Science & Engineering'
        ECE = 'ECE', 'Electronics & Communication'
        EEE = 'EEE', 'Electrical & Electronics'
        ME = 'ME', 'Mechanical Engineering'
        CV = 'CV', 'Civil Engineering'
        OTHER = 'OTHER', 'Other / Interdisciplinary'

    name = models.CharField(max_length=150, help_text="Full subject name, e.g. Machine Learning")
    code = models.CharField(max_length=20, help_text="Subject code or abbreviation, e,g. ML, DL, BAI701")
    slug = models.SlugField(max_length=160, unique=True, help_text="URL-Freindly identifier, e.g. machine-learning")
    description = models.TextField(blank=True, help_text="Sjort overview of what this subject covers.")
    order = models.PositiveSmallIntegerField(default=1, help_text="Display order on the homepage")
    branch = models.CharField(
        max_length=20,
        choices=Branch.choices,
        default=Branch.ALL,
        blank=True,
        help_text="Academic engineering branch/department"
    )
    semester = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="Academic semester (1 to 8), or null for common across semesters"
    )
    scheme = models.CharField(
        max_length=30,
        default="2022 Scheme",
        blank=True,
        help_text="Curriculum scheme (e.g. 2022 Scheme)"
    )

    class Meta:
        ordering = ['order', 'name']
        verbose_name = 'Subject'
        verbose_name_plural = "Subjects"

    def __str__(self):
        return f"{self.code} - {self.name}"

    @classmethod
    def infer_branch_and_semester(cls, code, name=""):
        """
        Intelligently infers engineering branch and semester number from course code (e.g. VTU scheme) or title.
        """
        import re
        clean_code = (code or "").strip().upper()
        clean_name = (name or "").strip().lower()

        branch = cls.Branch.ALL
        semester = None

        # 1. Infer Branch
        if any(prefix in clean_code for prefix in ('BAI', 'BAD', 'AIML', 'AI', 'ML')) or 'artificial intelligence' in clean_name or 'machine learning' in clean_name:
            branch = cls.Branch.AIML
        elif any(prefix in clean_code for prefix in ('BCS', 'CSE', '18CS', '21CS', '22CS')) or 'computer science' in clean_name:
            branch = cls.Branch.CSE
        elif any(prefix in clean_code for prefix in ('BIS', 'ISE', '18IS', '21IS')) or 'information science' in clean_name:
            branch = cls.Branch.ISE
        elif any(prefix in clean_code for prefix in ('BEC', 'ECE', '18EC', '21EC')) or 'electronics' in clean_name or 'communication' in clean_name:
            branch = cls.Branch.ECE
        elif any(prefix in clean_code for prefix in ('BEE', 'EEE', '18EE', '21EE')) or 'electrical' in clean_name:
            branch = cls.Branch.EEE
        elif any(prefix in clean_code for prefix in ('BME', 'MECH', '18ME', '21ME')) or 'mechanical' in clean_name or 'machining' in clean_name:
            branch = cls.Branch.ME
        elif any(prefix in clean_code for prefix in ('BCV', 'CIVIL', '18CV', '21CV')) or 'civil' in clean_name:
            branch = cls.Branch.CV
        elif clean_code.startswith('CS') or ' CS' in clean_code:
            branch = cls.Branch.CSE
        elif clean_code.startswith('EC') or ' EC' in clean_code:
            branch = cls.Branch.ECE
        elif clean_code.startswith('ME') or ' ME' in clean_code:
            branch = cls.Branch.ME

        # 2. Infer Semester
        m_vtu = re.search(r'[A-Za-z]+([1-8])\d{2}', clean_code)
        if m_vtu:
            semester = int(m_vtu.group(1))
        else:
            m_old_vtu = re.search(r'[A-Za-z]+([1-8])\d', clean_code)
            if m_old_vtu:
                semester = int(m_old_vtu.group(1))
            else:
                m_sem_name = re.search(r'(?:sem(?:ester)?|sem)\s*([1-8])|([1-8])(?:st|nd|rd|th)\s*sem', clean_name)
                if m_sem_name:
                    semester = int(m_sem_name.group(1) or m_sem_name.group(2))

        return branch, semester

    def save(self, *args, **kwargs):
        if not self.branch or self.branch == self.Branch.ALL or not self.semester:
            inferred_branch, inferred_sem = self.infer_branch_and_semester(self.code, self.name)
            if (not self.branch or self.branch == self.Branch.ALL) and inferred_branch != self.Branch.ALL:
                self.branch = inferred_branch
            if not self.semester and inferred_sem:
                self.semester = inferred_sem
        super().save(*args, **kwargs)

    @property
    def academic_tag(self):
        """Returns concise academic badge text, e.g. 'CSE · Sem 7' or 'Sem 7' or 'All Branches'."""
        parts = []
        if self.branch and self.branch != self.Branch.ALL:
            parts.append(self.branch)
        if self.semester:
            parts.append(f"Sem {self.semester}")
        elif not parts:
            return "General"
        return " · ".join(parts)


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

    @property
    def has_multiple_chapters(self):
        """Returns True if this module contains topics divided into distinct chapters or named chapters."""
        topics = list(self.topics.all())
        chapters = {t.chapter_number for t in topics if t.chapter_number}
        named_chapters = any(bool(t.chapter_title and t.chapter_title.strip()) for t in topics)
        return len(chapters) > 1 or named_chapters

    def get_chapters(self):
        """
        Groups topics belonging to this module by chapter, preserving order and calculating progress.
        Returns a list of dicts:
        [
            {
                'chapter_number': 1,
                'chapter_title': 'Key Management Fundamentals',
                'has_custom_title': True,
                'topics': [<Topic>, ...],
                'completed_topics': 2,
                'total_topics': 6,
                'progress_percent': 33,
            },
            ...
        ]
        """
        all_topics = list(self.topics.all())
        chapters_dict = {}
        for topic in all_topics:
            c_num = topic.chapter_number or 1
            if c_num not in chapters_dict:
                chapters_dict[c_num] = {
                    'chapter_number': c_num,
                    'chapter_title': topic.chapter_title or f"Chapter {c_num}",
                    'has_custom_title': bool(topic.chapter_title and topic.chapter_title.strip()),
                    'topics': [],
                    'completed_topics': 0,
                    'total_topics': 0,
                    'progress_percent': 0,
                }
            elif topic.chapter_title and not chapters_dict[c_num]['has_custom_title']:
                chapters_dict[c_num]['chapter_title'] = topic.chapter_title
                chapters_dict[c_num]['has_custom_title'] = True

            chapters_dict[c_num]['topics'].append(topic)
            chapters_dict[c_num]['total_topics'] += 1
            if hasattr(topic, 'progress') and topic.progress and topic.progress.status == 'COMPLETED':
                chapters_dict[c_num]['completed_topics'] += 1

        for c_data in chapters_dict.values():
            if c_data['total_topics'] > 0:
                c_data['progress_percent'] = int((c_data['completed_topics'] / c_data['total_topics']) * 100)

        return [chapters_dict[k] for k in sorted(chapters_dict.keys())]


class Topic(models.Model):
    module = models.ForeignKey(
        Module,
        on_delete=models.CASCADE,
        related_name="topics",
        help_text="The module this topic belongs to"
    )

    chapter_number = models.PositiveSmallIntegerField(
        default=1,
        help_text="Chapter sequence number within module (e.g. 1, 2)"
    )

    chapter_title = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Chapter title within module (e.g. Key Management Fundamentals)"
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
        ordering = ['module', 'chapter_number', 'order', 'title']
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


class ExamQuestion(models.Model):
    module = models.ForeignKey(
        Module,
        on_delete=models.CASCADE,
        related_name='exam_questions',
        help_text="The syllabus module this exam question belongs to"
    )
    topic = models.ForeignKey(
        Topic,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='exam_questions',
        help_text="Optional topic linked to this question"
    )
    question_text = models.TextField(help_text="The VTU exam question prompt")
    marks = models.PositiveSmallIntegerField(default=10, help_text="Marks weightage (e.g. 5, 6, 7, 8, 10)")
    frequency = models.PositiveSmallIntegerField(default=1, help_text="Number of times asked across VTU exams")
    years_asked = models.CharField(max_length=200, blank=True, default="", help_text="e.g. Jan 2023, July 2022")
    scheme = models.CharField(max_length=50, blank=True, default="2022 Scheme", help_text="Curriculum scheme")

    answer_blueprint = models.JSONField(
        null=True,
        blank=True,
        help_text="Cached structured answer format generated by AI"
    )
    blueprint_generated_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-frequency', '-marks', 'id']
        verbose_name = "Exam Question"
        verbose_name_plural = "Exam Questions"

    def __str__(self):
        try:
            return f"{self.module.subject.code} M{self.module.module_number} [{self.marks}M]: {self.question_text[:50]}"
        except Exception:
            return f"Exam Question #{self.pk}"

    @property
    def is_repeated(self):
        return self.frequency >= 2 or bool(self.years_asked and (',' in self.years_asked or ';' in self.years_asked))

    @property
    def is_frequent(self):
        return self.is_repeated



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
    branch = models.CharField(max_length=20, blank=True, default='', help_text="Target branch/department")
    semester = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Target semester (1-8)")

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

