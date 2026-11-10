import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studyhub.settings')
django.setup()

from django.contrib.auth import get_user_model

def create_admin_user():
    User = get_user_model()
    username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
    password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')
    email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@studyhub.com')

    if not username or not password:
        print("DJANGO_SUPERUSER_USERNAME or DJANGO_SUPERUSER_PASSWORD not set. Skipping automatic superuser creation.")
        return

    if User.objects.filter(username=username).exists():
        print(f"Superuser '{username}' already exists. No action needed.")
    else:
        print(f"Creating production superuser '{username}'...")
        User.objects.create_superuser(username=username, email=email, password=password)
        print(f"Superuser '{username}' successfully created!")

def cleanup_orphaned_seed_notes():
    """Removes the local development dummy note that does not exist on Cloudinary."""
    try:
        from curriculum.models import ModuleNote
        orphans = ModuleNote.objects.filter(file__contains='BAI701-module-1-textbook.pdf')
        if orphans.exists():
            count = orphans.count()
            orphans.delete()
            print(f"Removed {count} orphaned test note(s) from database.")
    except Exception as e:
        print(f"Cleanup note warning: {e}")

if __name__ == '__main__':
    create_admin_user()
    cleanup_orphaned_seed_notes()
