from django import template

register = template.Library()

@register.filter
def get_packet_count(piece, packet):
    return piece.get_packet_count(packet)