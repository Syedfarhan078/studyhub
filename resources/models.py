from django.db import models
from curriculum.models import Topic
# Create your models here.

class VideoResource(models.Model):
    topic = models.ForeignKey(
        Topic,
        on_delete=models.CASCADE,
        related_name='videos',
        help_text="The syllabus topic this video teaches"
    )

    youtube_video_id = models.CharField(
        max_length=60,
        help_text="11-character Youtube video ID (e.g. dQw4w8sjde)"
    )

    title = models.CharField(
        max_length=255,
        help_text="Title of the video from youtube"
    )

    channel_title = models.CharField(
        max_length=200,
        help_text="None of the YouTube channel/creator"
    )

    thumbnail_url = models.URLField(
        max_length=500,
        help_text="Direct-link to the video thumbnail image"
    )

    description = models.TextField(
        blank=True,
        help_text="Video Description snippet"
    )

    published_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Original publication date on YouTube"
    )

    cached_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when this video was fetched and cached in our database"
    )

    class Meta:
        ordering = ['-published_at']
        unique_together = ['topic', 'youtube_video_id']
        verbose_name = "Video Resource"
        verbose_name_plural = "Video Resources"

    def __str__(self):
        return f"{self.title} ({self.channel_title})"

    @property
    def embed_url(self):
        return f"https://www.youtube-nocookie.com/embed/{self.youtube_video_id}"
    
    @property
    def watch_url(self):
        return f"https://www.youtube.com/watch?v={self.youtube_video_id}"
    

class ArticleResource(models.Model):
    topic = models.ForeignKey(
        Topic,
        on_delete=models.CASCADE,
        related_name='articles',
        help_text="The syllabus topic this article covers"
    )
    title = models.CharField(
        max_length=300,
        help_text="Article headline or title"
    )
    url = models.URLField(
        max_length=600,
        help_text="Direct URL to the article"
    )
    source_name = models.CharField(
        max_length=120,
        help_text="Source publication (e.g. GeeksforGeeks, Towards Data Science, arXiv, IBM)"
    )
    snippet = models.TextField(
        blank=True,
        help_text="Short introductory summary or excerpt"
    )
    published_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Publication date, when available"
    )
    cached_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this article was discovered and stored in our database"
    )

    class Meta:
        ordering = ['-published_date', '-cached_at']
        unique_together = ['topic', 'url']
        verbose_name = "Article Resource"
        verbose_name_plural = "Article Resources"

    def __str__(self):
        return f"[{self.source_name}] {self.title}"