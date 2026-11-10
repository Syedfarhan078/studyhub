from django.contrib import admin
from .models import VideoResource, ArticleResource
# Register your models here.

@admin.register(VideoResource)
class VideoResourceAdmin(admin.ModelAdmin):
    list_display = ('title', 'channel_title', 'topic', 'published_at', 'cached_at')
    list_filter = ('topic__module__subject', 'topic')
    search_fields = ('title', 'channel_title', 'youtube_video_id')
    readonly_fields = ('cached_at',)


@admin.register(ArticleResource)
class ArticleResourceAdmin(admin.ModelAdmin):
    list_display = ('title', 'source_name', 'topic', 'published_date', 'cached_at')
    list_filter = ('source_name', 'topic__module__subject', 'topic')
    search_fields = ('title', 'snippet', 'source_name', 'url')
    readonly_fields = ('cached_at',)