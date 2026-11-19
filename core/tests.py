import json
import os
from django.test import TestCase
from django.urls import reverse
from django.conf import settings


class OfflinePWATests(TestCase):
    def test_manifest_file_valid_json(self):
        """Verifies that static/manifest.json exists and contains required PWA metadata."""
        manifest_path = os.path.join(settings.BASE_DIR, 'static', 'manifest.json')
        self.assertTrue(os.path.exists(manifest_path), "manifest.json should exist in static directory")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(data.get('short_name'), 'StudyHub')
        self.assertEqual(data.get('display'), 'standalone')
        self.assertEqual(data.get('start_url'), '/')
        self.assertEqual(data.get('background_color'), '#f3f0e8')
        self.assertTrue(len(data.get('icons', [])) > 0)

    def test_service_worker_endpoint(self):
        """Verifies /sw.js serves the root-scoped service worker with proper headers."""
        url = reverse('core:service_worker')
        res = self.client.get(url)

        self.assertEqual(res.status_code, 200)
        self.assertIn('application/javascript', res['Content-Type'])
        self.assertEqual(res.get('Service-Worker-Allowed'), '/')
        content = res.content.decode('utf-8')
        self.assertIn('studyhub-v1', content)
        self.assertIn('cache_pinned', content)
        self.assertIn('NOTES_CACHE', content)

    def test_offline_fallback_page(self):
        """Verifies /offline/ renders the dedicated offline classroom template."""
        url = reverse('core:offline')
        res = self.client.get(url)

        self.assertEqual(res.status_code, 200)
        self.assertTemplateUsed(res, 'core/offline.html')
        self.assertContains(res, "You're Studying Offline")
        self.assertContains(res, "Open My Desk")
        self.assertContains(res, "offlinePinnedList")

    def test_base_template_pwa_integration(self):
        """Verifies base template includes manifest link and SW registration."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'rel="manifest"')
        self.assertContains(res, 'navigator.serviceWorker.register')
        self.assertContains(res, 'offlineModeIndicator')
        self.assertContains(res, 'btnPwaInstall')
