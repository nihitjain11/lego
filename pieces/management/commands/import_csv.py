import csv
from django.core.management.base import BaseCommand
from pieces.models import LegoPiece

class Command(BaseCommand):
    help = 'Import LEGO pieces data from CSV file'

    def handle(self, *args, **options):
        csv_file = 'LEGO Sorting Sheet - Sheet2.csv'
        
        # Clear existing data
        LegoPiece.objects.all().delete()
        
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            pieces = []
            
            for row in reader:
                # Create piece with basic data
                piece = LegoPiece(
                    page=int(row['Page']),
                    color=row['Color'],
                    shape=row['Shape'],
                    total_count=int(row['Count']),
                    remaining_count=int(row['Remain']),
                )
                
                # Initialize packet counts
                packet_counts = {}
                for i in range(46):  # 0-45
                    count = row.get(f'#{i}', '')
                    if count and count.strip():
                        packet_counts[str(i)] = int(count)
                
                piece.packet_counts = packet_counts
                pieces.append(piece)
            
            # Bulk create all pieces
            LegoPiece.objects.bulk_create(pieces)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully imported {len(pieces)} LEGO pieces')
            )