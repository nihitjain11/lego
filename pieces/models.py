from django.db import models

# Create your models here.
class LegoPiece(models.Model):
    page = models.IntegerField()
    color = models.CharField(max_length=50)
    shape = models.CharField(max_length=50)
    total_count = models.IntegerField()
    remaining_count = models.IntegerField()
    image = models.ImageField(upload_to='piece_images/', null=True, blank=True)
    packet_counts = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.color} {self.shape} (Page {self.page})"

    class Meta:
        ordering = ['page', 'color', 'shape']

    def get_packet_count(self, packet):
        return int(self.packet_counts.get(str(packet), 0))

    def update_remaining_count(self):
        assigned_count = sum(int(count) for count in self.packet_counts.values())
        self.remaining_count = self.total_count - assigned_count
        self.save(update_fields=['remaining_count'])

    def set_packet_count(self, packet, count):
        count = max(0, min(count, self.total_count))  # Ensure count is between 0 and total_count
        self.packet_counts[str(packet)] = count
        self.update_remaining_count()

    def increment_packet(self, packet):
        current = self.get_packet_count(packet)
        if current < self.total_count and self.remaining_count > 0:
            self.set_packet_count(packet, current + 1)
            return True
        return False

    def decrement_packet(self, packet):
        current = self.get_packet_count(packet)
        if current > 0:
            self.set_packet_count(packet, current - 1)
            return True
        return False

    def get_static_image_url(self):
        """Returns the URL of the static image based on the piece number (shape)"""
        from django.templatetags.static import static
        return static(f'pieces/{self.shape}.png')

    def has_static_image(self):
        """Check if a static image exists for this piece"""
        from django.contrib.staticfiles.storage import staticfiles_storage
        return staticfiles_storage.exists(f'pieces/{self.shape}.png')
