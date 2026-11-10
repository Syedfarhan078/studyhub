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

if __name__ == '__main__':
    create_admin_user()
