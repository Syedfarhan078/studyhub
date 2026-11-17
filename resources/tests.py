import json
from unittest.mock import patch, MagicMock
from django.test import TestCase
from curriculum.models import Subject, Module, Topic, TopicKeyword
from resources.models import VideoResource, ArticleResource
from resources.services.youtube import (
    clean_topic_title,
    build_educational_youtube_query,
    get_or_fetch_videos_for_topic,
)
from resources.services.articles import (
    clean_html_snippet,
    clean_topic_title as clean_article_topic_title,
    fetch_wikipedia_articles,
    get_or_fetch_articles_for_topic,
)


class YouTubeServiceTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Network Security and Cryptography",
            code="CS701",
            slug="network-security-cs701",
            description="Core engineering course"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Classical Encryption Techniques"
        )
        self.topic = Topic.objects.create(
            module=self.module,
            title="1.1 Symmetric cipher model",
            slug="symmetric-cipher-model"
        )

    def test_clean_topic_title(self):
        self.assertEqual(clean_topic_title("1.1 Symmetric cipher model"), "Symmetric cipher model")
        self.assertEqual(clean_topic_title("Topic 2: Substitution ciphers"), "Substitution ciphers")
        self.assertEqual(clean_topic_title("• Transposition ciphers"), "Transposition ciphers")

    def test_build_educational_youtube_query_with_subject_context(self):
        query = build_educational_youtube_query(self.topic)
        self.assertIn("Symmetric cipher model", query)
        self.assertIn("Network Security and Cryptography", query)
        self.assertIn("lecture", query)

    def test_build_educational_youtube_query_uses_academic_keyword(self):
        TopicKeyword.objects.create(
            topic=self.topic,
            keyword="Symmetric cipher model NPTEL engineering lecture"
        )
        query = build_educational_youtube_query(self.topic)
        self.assertEqual(query, "Symmetric cipher model NPTEL engineering lecture")

    @patch('resources.services.youtube.requests.get')
    def test_get_or_fetch_videos_saves_resources(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": {"videoId": "vid12345678"},
                    "snippet": {
                        "title": "Symmetric Cipher Model &amp; Cryptography",
                        "channelTitle": "NPTEL IIT",
                        "description": "Full lecture on symmetric key encryption.",
                        "thumbnails": {"high": {"url": "https://img.youtube.com/vi/vid12345678/hqdefault.jpg"}},
                        "publishedAt": "2026-01-15T10:00:00Z"
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        with patch('resources.services.youtube.config', return_value='fake_youtube_key'):
            videos = get_or_fetch_videos_for_topic(self.topic, force_refresh=True)

        self.assertEqual(videos.count(), 1)
        video = videos.first()
        self.assertEqual(video.youtube_video_id, "vid12345678")
        self.assertEqual(video.title, "Symmetric Cipher Model & Cryptography")
        self.assertEqual(video.channel_title, "NPTEL IIT")


class ArticleServiceTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Operating Systems",
            code="CS403",
            slug="operating-systems-cs403"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=2,
            title="Process Synchronization"
        )
        self.topic = Topic.objects.create(
            module=self.module,
            title="Critical Section Problem",
            slug="critical-section-problem"
        )

    def test_clean_html_snippet(self):
        raw = "<span>Peterson's <b>Algorithm</b> &amp; Mutex</span>"
        cleaned = clean_html_snippet(raw)
        self.assertEqual(cleaned, "Peterson's Algorithm & Mutex")

    @patch('resources.services.articles.requests.get')
    def test_fetch_wikipedia_articles(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Critical section",
                        "snippet": "In concurrent programming, concurrent accesses to shared resources can lead to unexpected or erroneous behavior."
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        count = fetch_wikipedia_articles(self.topic, max_results=1)
        self.assertEqual(count, 1)

        articles = self.topic.articles.all()
        self.assertEqual(articles.count(), 1)
        art = articles.first()
        self.assertEqual(art.title, "Critical section")
        self.assertEqual(art.source_name, "Wikipedia Academic")
        self.assertIn("https://en.wikipedia.org/wiki/Critical_section", art.url)
