# Generated by Django 4.2.5 on 2023-12-05 19:43

from django.conf import settings
import django.core.serializers.json
from django.db import migrations, models
import django.db.models.deletion
import stock_news_service.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user_service', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='GDELTCall',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('call_time', models.DateTimeField(auto_now_add=True)),
                ('interval', models.IntegerField()),
                ('top_n', models.IntegerField()),
                ('config', models.JSONField(encoder=django.core.serializers.json.DjangoJSONEncoder, null=True)),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='gdelt_calls', to='user_service.stock')),
            ],
        ),
        migrations.CreateModel(
            name='GDELTEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField()),
                ('top_articles', models.TextField()),
                ('top_themes', models.TextField()),
                ('top_persons', models.TextField()),
                ('top_orgs', models.TextField()),
                ('top_locs', models.TextField()),
                ('cluster_health', models.JSONField(default=list, encoder=django.core.serializers.json.DjangoJSONEncoder)),
                ('median_date', models.DateTimeField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('event_window', models.IntegerField()),
                ('mean_embedding', models.JSONField(default=stock_news_service.models.default_float_list, encoder=django.core.serializers.json.DjangoJSONEncoder)),
                ('irelevant', models.BooleanField(default=False)),
                ('gdelt_call', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='gdelt_events_call', to='stock_news_service.gdeltcall')),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='gdelt_events', to='user_service.stock')),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='gdelt_events', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='GDELTEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.TextField(null=True)),
                ('themes', models.TextField(null=True)),
                ('tone', models.TextField(null=True)),
                ('locations', models.TextField(null=True)),
                ('persons', models.TextField(null=True)),
                ('organizations', models.TextField(null=True)),
                ('document_identifier', models.TextField(null=True)),
                ('all_names', models.TextField(null=True)),
                ('amounts', models.TextField(null=True)),
                ('gdelt_call', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='gdelt_entries', to='stock_news_service.gdeltcall')),
            ],
        ),
    ]