from django.shortcuts import render, get_object_or_404
from django.contrib import messages
from django.db.models import Case, When, Value, IntegerField
from curriculum.models import Topic
from .services.youtube import get_or_fetch_videos_for_topic
from .services.articles import get_or_fetch_articles_for_topic


def topic_videos(request, topic_id):
    """
    Renders the dedicated YouTube learning platform for a specific topic.
    Supports a ?refresh=1 query parameter to bypass cache and query fresh videos.
    """
    topic = get_object_or_404(
        Topic.objects.select_related('module__subject').prefetch_related('keywords'),
        id=topic_id
    )

    force_refresh = request.GET.get('refresh') == '1'
    videos = get_or_fetch_videos_for_topic(topic, force_refresh=force_refresh)

    if force_refresh:
        messages.success(request, f"Refreshed YouTube video results for {topic.title}!")

    context = {
        'topic': topic,
        'subject': topic.module.subject,
        'module': topic.module,
        'videos': videos,
    }
    return render(request, 'resources/video_list.html', context)


def topic_articles(request, topic_id):
    """
    Renders the dedicated article discovery feed for a specific syllabus topic.
    Supports a ?refresh=1 query parameter to fetch fresh articles.
    """
    topic = get_object_or_404(
        Topic.objects.select_related('module__subject').prefetch_related('keywords'),
        id=topic_id
    )

    force_refresh = request.GET.get('refresh') == '1'
    articles = get_or_fetch_articles_for_topic(topic, force_refresh=force_refresh)

    articles = articles.annotate(
        priority=Case(
            When(source_name__icontains='GeeksforGeeks', then=Value(1)),
            When(source_name__icontains='W3Schools', then=Value(2)),
            default=Value(3),
            output_field=IntegerField()
        )
    ).order_by('priority', '-published_date')

    if force_refresh:
        messages.success(request, f"Refreshed article discovery feed for {topic.title}!")

    context = {
        'topic': topic,
        'subject': topic.module.subject,
        'module': topic.module,
        'articles': articles,
    }
    return render(request, 'resources/article_list.html', context)