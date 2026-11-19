import json
import os
from django.conf import settings
from django.http import HttpResponse, Http404, JsonResponse
from django.shortcuts import render
from django.db.models import Count
from curriculum.models import Subject

# Create your views here.

def home(request):
    """
    Renders the StudyHub homepage supporting the Hybrid Academic model:
    - Personal 'My Desk' containing only the student's pinned subjects (zero accounts/login).
    - 'University Catalog' searchable and filterable by engineering branch and semester.
    - Synchronizes with long-lived client cookies and session.
    - Supplies subject pacing metadata for the Exam Day Countdown & Study Schedule engine.
    """
    pinned_set = set(request.session.get('studyhub_pinned', []))
    cookie_pinned = request.COOKIES.get('studyhub_pinned', '')
    if cookie_pinned:
        for s in cookie_pinned.split(','):
            s_clean = s.strip()
            if s_clean:
                pinned_set.add(s_clean)

    subjects = list(
        Subject.objects.annotate(
            module_count=Count('modules')
        ).prefetch_related(
            'modules__topics'
        ).order_by('order', 'name')
    )

    for s in subjects:
        s.is_pinned = s.slug in pinned_set

    pinned_subjects = [s for s in subjects if s.is_pinned]

    branch_choices = [
        ('ALL', 'All Branches'),
        ('CSE', 'Computer Science (CSE)'),
        ('AIML', 'AI & Machine Learning (AIML)'),
        ('ISE', 'Information Science (ISE)'),
        ('ECE', 'Electronics & Comm (ECE)'),
        ('EEE', 'Electrical & Electronics (EEE)'),
        ('ME', 'Mechanical Engineering (ME)'),
        ('CV', 'Civil Engineering (CV)'),
        ('OTHER', 'Other / Interdisciplinary'),
    ]
    semesters = [1, 2, 3, 4, 5, 6, 7, 8]

    # Pre-build pacing metadata for all subjects so countdown & schedule calculations run instantly on client
    pacing_catalog = {}
    for s in subjects:
        mod_list = []
        for m in s.modules.all().order_by('module_number'):
            m_topics = list(m.topics.all().order_by('order'))
            hy_topics = [t.title for t in m_topics if getattr(t, 'is_high_yield', False)]
            if not hy_topics and m_topics:
                hy_topics = [t.title for t in m_topics[:2]]
            mod_list.append({
                'module_number': m.module_number,
                'title': m.title,
                'total_topics': len(m_topics),
                'high_yield_topics': hy_topics,
            })
        pacing_catalog[s.slug] = {
            'name': s.name,
            'code': s.code,
            'slug': s.slug,
            'academic_tag': s.academic_tag,
            'modules': mod_list,
        }

    context = {
        'subjects': subjects,
        'pinned_subjects': pinned_subjects,
        'pinned_count': len(pinned_subjects),
        'branches': branch_choices,
        'semesters': semesters,
        'pacing_catalog_json': json.dumps(pacing_catalog),
    }

    return render(request, 'core/index.html', context)


def service_worker(request):
    """
    Serves the root-scoped PWA service worker with appropriate headers.
    """
    sw_path = os.path.join(settings.BASE_DIR, 'static', 'js', 'sw.js')
    if not os.path.exists(sw_path):
        raise Http404("Service worker file not found.")
    with open(sw_path, 'r', encoding='utf-8') as f:
        content = f.read()
    response = HttpResponse(content, content_type='application/javascript')
    response['Service-Worker-Allowed'] = '/'
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


def offline_fallback(request):
    """
    Renders a friendly offline page if a student navigates to an uncached page while completely offline.
    """
    return render(request, 'core/offline.html')


def robots_txt(request):
    """
    Dynamically generates robots.txt, directing web crawlers to the sitemap
    and disallowing private or mutating paths.
    """
    sitemap_url = request.build_absolute_uri('/sitemap.xml')
    lines = [
        "User-agent: *",
        "Allow: /",
        "Disallow: /admin/",
        "Disallow: /subjects/generate-ai/",
        "Disallow: /subjects/generation-status/",
        "Disallow: /subjects/toggle-pin/",
        f"Sitemap: {sitemap_url}",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


