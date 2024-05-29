# Generated by Django 4.2.5 on 2023-12-05 19:43

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user_service', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Alert',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frequency', models.CharField(choices=[('D', 'Daily'), ('W', 'Weekly')], default='D', max_length=1)),
                ('last_sent', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='AlertStock',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alert', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='alert_stocks', to='alert_service.alert')),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='user_service.stock')),
            ],
            options={
                'unique_together': {('alert', 'stock')},
            },
        ),
        migrations.CreateModel(
            name='AreaOfInterest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=255)),
                ('alert_stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='areas_of_interest', to='alert_service.alertstock')),
            ],
        ),
    ]