from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.conf import settings
import os
from pathlib import Path

class Command(BaseCommand):
    help = 'Updates static files and reports any missing images for pieces'

    def handle(self, *args, **options):
        # Run collectstatic
        self.stdout.write('Collecting static files...')
        call_command('collectstatic', '--noinput')
        
        # Get list of all PNG files in the static/pieces directory
        pieces_dir = Path(settings.STATICFILES_DIRS[0]) / 'pieces'
        if pieces_dir.exists():
            self.stdout.write(self.style.SUCCESS(f'Found {len(list(pieces_dir.glob("*.png")))} piece images'))
        else:
            self.stdout.write(self.style.WARNING('No pieces directory found in static folder'))