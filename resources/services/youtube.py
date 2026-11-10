import html
import logging
import requests
from django.utils.dateparse import parse_datetime
from decouple import config
from resources.models import VideoResource

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def get_or_fetch_videos_for_topic(topic, force_refresh=False):
    """
    Retrieves educational videos for a specific syllabus topic.
    
    1. Checks if videos are already cached in PostgreSQL.
    2. If cached and force_refresh is False, returns the cached videos (0 API quota used!).
    3. If not cached (or force_refresh is True), queries the YouTube Data API v3
       using the topic's search keywords, stores results in DB, and returns them.
    """
    # Check cache first
    cached_videos = topic.videos.all()
    if cached_videos.exists() and not force_refresh:
        return cached_videos

    api_key = config('YOUTUBE_API_KEY', default=None)
    if not api_key or api_key == 'your_key_will_go_here_later':
        logger.warning("YouTube API key is missing or not configured in .env.")
        return cached_videos

    # Formulate a search query using the topic title and up to 3 keywords
    # Pick the most relevant search phrase (prefer the first keyword, fallback to clean topic title)
    first_keyword = topic.keywords.values_list('keyword', flat=True).first()
    if first_keyword:
        query = first_keyword
    else:
        query = f"{topic.title} tutorial"

    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'videoEmbeddable': 'true',
        'maxResults': 9,
        'relevanceLanguage': 'en',
        'key': api_key,
    }

    print(f"\n[DEBUG] Querying YouTube for: '{query}'")

    try:
        response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=8)
        print(f"[DEBUG] YouTube API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[DEBUG] YouTube Error Response: {response.text}")
        
        # Handle Quota Limit or Forbidden
        if response.status_code == 403:
            logger.error("YouTube API quota exceeded or key invalid (403 Forbidden).")
            return cached_videos

        response.raise_for_status()
        data = response.json()

        items = data.get('items', [])
        if not items:
            return cached_videos

        # If force refreshing, clear older cached videos for this topic
        if force_refresh:
            topic.videos.all().delete()

        # Parse and save new video records into PostgreSQL
        for item in items:
            video_id = item.get('id', {}).get('videoId')
            snippet = item.get('snippet', {})

            if not video_id:
                continue

            # YouTube API returns HTML-encoded entities like &quot; and &#39;
            clean_title = html.unescape(snippet.get('title', ''))
            clean_description = html.unescape(snippet.get('description', ''))
            clean_channel = html.unescape(snippet.get('channelTitle', ''))

            thumbnails = snippet.get('thumbnails', {})
            # Prefer high quality thumbnail, fallback to medium or default
            thumb_url = (
                thumbnails.get('high', {}).get('url') or
                thumbnails.get('medium', {}).get('url') or
                thumbnails.get('default', {}).get('url') or ''
            )

            published_at = parse_datetime(snippet.get('publishedAt', ''))

            VideoResource.objects.update_or_create(
                topic=topic,
                youtube_video_id=video_id,
                defaults={
                    'title': clean_title,
                    'channel_title': clean_channel,
                    'thumbnail_url': thumb_url,
                    'description': clean_description,
                    'published_at': published_at,
                }
            )

        return topic.videos.all()

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while connecting to YouTube API: {e}")
        return cached_videos