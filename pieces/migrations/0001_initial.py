# Generated by Django 5.2.1 on 2025-05-26 15:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LegoPiece',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page', models.IntegerField()),
                ('color', models.CharField(max_length=50)),
                ('shape', models.CharField(max_length=50)),
                ('total_count', models.IntegerField()),
                ('remaining_count', models.IntegerField()),
                ('image', models.ImageField(blank=True, null=True, upload_to='piece_images/')),
                ('packet_counts', models.JSONField(default=dict)),
            ],
            options={
                'ordering': ['page', 'color', 'shape'],
            },
        ),
    ]
