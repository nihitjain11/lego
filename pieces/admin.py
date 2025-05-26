from django.contrib import admin
from .models import LegoPiece

@admin.register(LegoPiece)
class LegoPieceAdmin(admin.ModelAdmin):
    list_display = ['page', 'color', 'shape', 'total_count', 'remaining_count']
    search_fields = ['color', 'shape']
    list_filter = ['page', 'color']
