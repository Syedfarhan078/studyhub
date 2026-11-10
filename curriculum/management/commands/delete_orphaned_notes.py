"""
One-time management command to delete the orphaned dummy seed note from production.
Run via: python manage.py delete_orphaned_notes
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Deletes orphaned/dummy module notes that have no real file on Cloudinary'

    def handle(self, *args, **options):
        from curriculum.models import ModuleNote

        # List all notes and show what we find
        all_notes = ModuleNote.objects.all()
        self.stdout.write(f'Total notes in database: {all_notes.count()}')

        for note in all_notes:
            self.stdout.write(f'  Note #{note.pk}: title="{note.title}", file="{note.file.name}"')

        # Delete any notes whose file path contains known dummy/test patterns
        dummy_patterns = [
            'BAI701-module-1-textbook',
            'test.pdf',
        ]

        deleted_count = 0
        for pattern in dummy_patterns:
            qs = ModuleNote.objects.filter(file__icontains=pattern)
            if qs.exists():
                for n in qs:
                    self.stdout.write(self.style.WARNING(
                        f'Deleting orphaned note #{n.pk}: "{n.title}" ({n.file.name})'
                    ))
                count = qs.count()
                qs.delete()
                deleted_count += count

        if deleted_count:
            self.stdout.write(self.style.SUCCESS(f'Successfully deleted {deleted_count} orphaned note(s).'))
        else:
            self.stdout.write(self.style.SUCCESS('No orphaned notes found. Database is clean.'))
