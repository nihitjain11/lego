from django.urls import path
from . import views

urlpatterns = [
    path('', views.piece_list, name='piece_list'),
    path('pieces/<int:piece_id>/update_count/', views.update_piece_count, name='update_count'),
    path('pieces/<int:piece_id>/', views.piece_detail, name='piece_detail'),
    path('packets/', views.packet_list, name='packet_list'),
    path('import-csv/', views.import_csv, name='import_csv'),
]