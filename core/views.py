from django.shortcuts import render
from django.db.models import Count
from django.http import HttpResponse
from django.core.management import call_command
from curriculum.models import Subject

def home(request):
    subjects = Subject.objects.annotate(
        module_count=Count('modules')
    ).order_by('order', 'name')

    # If database is fresh (0 subjects on new deployment), auto-populate once from curriculum_data.json
    if not subjects.exists():
        try:
            call_command('migrate', interactive=False)
            call_command('loaddata', 'curriculum_data.json')
            subjects = Subject.objects.annotate(
                module_count=Count('modules')
            ).order_by('order', 'name')
        except Exception as e:
            print(f"Auto-seed exception: {e}")

    context = {
        'subjects': subjects,
    }

    return render(request, 'core/index.html', context)

def seed_data(request):
    """Manual seed trigger endpoint for initial production deployment."""
    try:
        call_command('migrate', interactive=False)
        call_command('loaddata', 'curriculum_data.json')
        count = Subject.objects.count()
        return HttpResponse(
            f"<div style='font-family: sans-serif; text-align: center; padding: 40px;'>"
            f"<h2 style='color: #10b981;'>Success! Loaded {count} subjects into database.</h2>"
            f"<p><a href='/' style='color: #3b82f6; font-size: 18px;'>Go to Curriculum Dashboard &rarr;</a></p>"
            f"</div>"
        )
    except Exception as e:
        return HttpResponse(
            f"<div style='font-family: sans-serif; padding: 40px; color: #ef4444;'>"
            f"<h2>Error seeding data</h2><pre>{e}</pre>"
            f"</div>"
        )
