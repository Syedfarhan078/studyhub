import json
import os
from django.test import TestCase
from django.urls import reverse
from django.conf import settings
from curriculum.models import Subject


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


class SeoAndSitemapTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Computer Networks",
            code="BCS502",
            slug="computer-networks",
            branch=Subject.Branch.CSE,
            semester=5,
            order=1
        )

    def test_subject_get_absolute_url(self):
        self.assertEqual(self.subject.get_absolute_url(), '/subjects/computer-networks/')

    def test_robots_txt(self):
        res = self.client.get('/robots.txt')
        self.assertEqual(res.status_code, 200)
        self.assertIn('text/plain', res['Content-Type'])
        content = res.content.decode('utf-8')
        self.assertIn('User-agent: *', content)
        self.assertIn('Allow: /', content)
        self.assertIn('Disallow: /admin/', content)
        self.assertIn('Disallow: /subjects/generate-ai/', content)
        self.assertIn('Sitemap: http://testserver/sitemap.xml', content)

    def test_sitemap_xml(self):
        res = self.client.get('/sitemap.xml')
        self.assertEqual(res.status_code, 200)
        self.assertIn('xml', res['Content-Type'])
        content = res.content.decode('utf-8')
        self.assertIn('<urlset', content)
        self.assertIn('/subjects/computer-networks/', content)
        self.assertIn('http://testserver/', content)

    def test_base_template_seo_and_gsc(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'google-site-verification')
        self.assertContains(res, 'zhm6-a3OqA9YctRN0cR_CtmOLZzKYo_9mkU60BlLNUM')
        self.assertContains(res, 'name="robots" content="index, follow"')
        self.assertContains(res, 'All rights reserved 2026')
        self.assertContains(res, '2025 Scheme')

    def test_admin_button_visibility(self):
        # Anonymous student does not see admin links
        res_anon = self.client.get('/')
        self.assertNotContains(res_anon, 'href="/admin/"')
        self.assertContains(res_anon, 'VTU All Schemes')

        # Staff user sees admin link
        from django.contrib.auth import get_user_model
        User = get_user_model()
        admin_user = User.objects.create_superuser('admin_tester', 'admin@example.com', 'password123')
        self.client.force_login(admin_user)
        res_staff = self.client.get('/')
        self.assertContains(res_staff, 'href="/admin/"')
