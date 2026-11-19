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


class ExamPacingDashboardTests(TestCase):
    def setUp(self):
        from curriculum.models import Module, Topic
        self.subject = Subject.objects.create(
            name="Operating Systems",
            code="BCS402",
            slug="operating-systems",
            branch=Subject.Branch.CSE,
            semester=4,
            order=1
        )
        self.m1 = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Introduction to OS & Process Management"
        )
        self.m2 = Module.objects.create(
            subject=self.subject,
            module_number=2,
            title="Threads & CPU Scheduling"
        )
        self.t1 = Topic.objects.create(
            module=self.m1,
            title="Process Control Block & State Transitions",
            slug="pcb-state-transitions",
            order=1,
            is_high_yield=True
        )
        self.t2 = Topic.objects.create(
            module=self.m1,
            title="Inter-Process Communication (IPC)",
            slug="ipc",
            order=2,
            is_high_yield=True
        )
        self.t3 = Topic.objects.create(
            module=self.m2,
            title="Multithreading Models & Scheduling Algorithms",
            slug="multithreading-models",
            order=1,
            is_high_yield=True
        )

    def test_home_view_pacing_catalog_context(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn('pacing_catalog_json', res.context)
        pacing_data = json.loads(res.context['pacing_catalog_json'])
        self.assertIn(self.subject.slug, pacing_data)
        subj_entry = pacing_data[self.subject.slug]
        self.assertEqual(subj_entry['name'], 'Operating Systems')
        self.assertEqual(subj_entry['code'], 'BCS402')
        self.assertEqual(len(subj_entry['modules']), 2)
        m1_data = subj_entry['modules'][0]
        self.assertEqual(m1_data['module_number'], 1)
        self.assertEqual(m1_data['total_topics'], 2)
        self.assertIn('Process Control Block & State Transitions', m1_data['high_yield_topics'])

    def test_home_template_exam_pacing_elements(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'activeExamSprintBanner')
        self.assertContains(res, 'desk-exam-pacing-slot')
        self.assertContains(res, 'examPacingModal')
        self.assertContains(res, 'pacingSubjectSelect')
        self.assertContains(res, 'pacingDatetimeInput')
        self.assertContains(res, 'pacingHoursRange')
        self.assertContains(res, 'pacingScheduleCardsContainer')
        self.assertContains(res, 'The 4-Hour Module Trap Warning')
        self.assertContains(res, 'Last 60-Minute Visual Memory Sprint')

    def test_subject_detail_exam_countdown_button(self):
        res = self.client.get(f'/subjects/{self.subject.slug}/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'btnSubjectExamCountdown')
        self.assertContains(res, f'href="/?pacing={self.subject.slug}"')
        self.assertContains(res, 'subjectCountdownLabel')
