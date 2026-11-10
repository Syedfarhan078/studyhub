"""
WSGI config for studyhub project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studyhub.settings')

application = get_wsgi_application()

# When running on Vercel serverless lambda, auto-migrate and seed curriculum data to Neon Cloud DB
if os.environ.get('VERCEL'):
    try:
        from django.core.management import call_command
        call_command('migrate', interactive=False)
        from curriculum.models import Subject
        if not Subject.objects.exists():
            call_command('loaddata', 'curriculum_data.json')
    except Exception as e:
        print(f"Vercel startup note: {e}")

app = application
