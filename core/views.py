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

    context = {
        'subjects': subjects,
        'pinned_subjects': pinned_subjects,
        'pinned_count': len(pinned_subjects),
        'branches': branch_choices,
        'semesters': semesters,
    }

    return render(request, 'core/index.html', context)
