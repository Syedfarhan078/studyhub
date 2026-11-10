from django.shortcuts import render
from django.db.models import Count
from curriculum.models import Subject

# Create your views here.

def home(request):
    subjects = Subject.objects.annotate(
        module_count=Count('modules')
    ).order_by('order', 'name')

    context = {
        'subjects': subjects,
    }

    return render(request, 'core/index.html', context)