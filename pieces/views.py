from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Q
from django.views.decorators.csrf import ensure_csrf_cookie
from .models import LegoPiece
import csv
from io import TextIOWrapper
from django.db import transaction
from django.contrib import messages

@ensure_csrf_cookie
def piece_list(request):
    # Get filter parameters
    search_query = request.GET.get('q', '')
    color_filter = request.GET.get('color', '')
    page_filter = request.GET.get('page', '')

    pieces = LegoPiece.objects.all()

    # Apply filters
    if search_query:
        pieces = pieces.filter(
            Q(shape__icontains=search_query) |
            Q(color__icontains=search_query)
        )
    if color_filter:
        pieces = pieces.filter(color__iexact=color_filter)
    if page_filter.strip():  # Only filter if page is not empty
        try:
            page_number = int(page_filter)
            pieces = pieces.filter(page=page_number)
        except ValueError:
            pass  # Invalid page number, ignore the filter

    # Get unique colors for filter dropdown
    colors = LegoPiece.objects.values_list('color', flat=True).distinct().order_by('color')
    
    # Create a list of packet numbers for iteration
    packet_range = range(46)  # 0 to 45 inclusive

    return render(request, 'pieces/piece_list.html', {
        'pieces': pieces,
        'colors': colors,
        'search_query': search_query,
        'color_filter': color_filter,
        'page_filter': page_filter,
        'packet_range': packet_range,
    })

def update_piece_count(request, piece_id):
    if request.method == 'POST':
        piece = get_object_or_404(LegoPiece, id=piece_id)
        packet = request.POST.get('packet')
        action = request.POST.get('action')

        if action == 'increment':
            success = piece.increment_packet(packet)
        else:
            success = piece.decrement_packet(packet)

        # Recalculate remaining count
        assigned_count = sum(piece.packet_counts.values())
        piece.remaining_count = piece.total_count - assigned_count
        piece.save()

        return JsonResponse({
            'success': success,
            'new_count': piece.get_packet_count(packet),
            'remaining': piece.remaining_count
        })

    return JsonResponse({'success': False}, status=400)

def piece_detail(request, piece_id):
    piece = get_object_or_404(LegoPiece, id=piece_id)
    if request.method == 'POST' and request.FILES.get('image'):
        piece.image = request.FILES['image']
        piece.save()
    
    return render(request, 'pieces/piece_detail.html', {'piece': piece})

def packet_list(request):
    packet_number = request.GET.get('packet', '')
    pieces_in_packet = []
    
    if packet_number.strip():
        # Get all pieces that have a count > 0 for this packet
        all_pieces = LegoPiece.objects.all()
        pieces_in_packet = [
            {
                'piece': piece,
                'count': piece.get_packet_count(packet_number)
            }
            for piece in all_pieces
            if piece.get_packet_count(packet_number) > 0
        ]
    
    # Get all packet numbers that have at least one piece
    packets_with_pieces = set()
    for piece in LegoPiece.objects.all():
        for packet, count in piece.packet_counts.items():
            if int(count) > 0:
                packets_with_pieces.add(int(packet))
    
    packets_with_pieces = sorted(list(packets_with_pieces))
    
    return render(request, 'pieces/packet_list.html', {
        'packets_with_pieces': packets_with_pieces,
        'selected_packet': packet_number,
        'pieces_in_packet': pieces_in_packet
    })

def import_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')
        try:
            with transaction.atomic():
                reader = csv.DictReader(csv_file)
                updates = 0
                creates = 0
                
                for row in reader:
                    piece, created = LegoPiece.objects.update_or_create(
                        shape=row['shape'],
                        color=row['color'],
                        defaults={
                            'page': int(row['page']),
                            'total_count': int(row['total_count']),
                            'remaining_count': int(row.get('remaining_count', row['total_count'])),
                        }
                    )
                    if created:
                        creates += 1
                    else:
                        updates += 1
                
                messages.success(
                    request, 
                    f'Successfully processed CSV: {creates} new pieces created, {updates} pieces updated.'
                )
                return redirect('piece_list')
                
        except Exception as e:
            messages.error(request, f'Error processing CSV: {str(e)}')
            transaction.rollback()
            
    return render(request, 'pieces/import_csv.html')
