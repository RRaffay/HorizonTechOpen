from django.db import models
from user_service.models import Stock
from django.contrib.auth.models import User
from django.core.serializers.json import DjangoJSONEncoder
import random


class GDELTCall(models.Model):
    call_time = models.DateTimeField(auto_now_add=True)
    interval = models.IntegerField()
    top_n = models.IntegerField()
    stock = models.ForeignKey(
        Stock, on_delete=models.CASCADE, related_name="gdelt_calls"
    )
    # Add a new nullable JSONField to store the configuration.
    config = models.JSONField(null=True, encoder=DjangoJSONEncoder)

    def __str__(self):
        return f"API Call at {self.call_time} for Stock {self.stock}"


class GDELTEntry(models.Model):
    date = models.TextField(null=True)
    themes = models.TextField(null=True)
    tone = models.TextField(null=True)
    locations = models.TextField(null=True)
    persons = models.TextField(null=True)
    organizations = models.TextField(null=True)
    document_identifier = models.TextField(null=True)
    all_names = models.TextField(null=True)
    amounts = models.TextField(null=True)
    gdelt_call = models.ForeignKey(
        GDELTCall, on_delete=models.CASCADE, related_name="gdelt_entries"
    )

    def __str__(self):
        return self.document_identifier


def default_float_list():
    # Should be set to default to embedding size of text-ada-02
    return [random.uniform(-1, 1) for _ in range(100)]


class GDELTEvent(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="gdelt_events", null=True
    )  # Link to a user
    stock = models.ForeignKey(
        Stock, on_delete=models.CASCADE, related_name="gdelt_events"
    )
    gdelt_call = models.ForeignKey(
        GDELTCall, on_delete=models.CASCADE, related_name="gdelt_events_call"
    )
    cluster_id = models.IntegerField()  # Cluster ID
    top_articles = (
        models.TextField()
    )  # List of top articles for the event, serialized as a string
    top_themes = models.TextField()
    top_persons = models.TextField()
    top_orgs = models.TextField()
    top_locs = models.TextField()
    # The health is in the following format: [size_score, cohesiveness_score, time_variance]
    cluster_health = models.JSONField(default=list, encoder=DjangoJSONEncoder)
    median_date = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    event_window = models.IntegerField()
    mean_embedding = models.JSONField(
        default=default_float_list, encoder=DjangoJSONEncoder
    )
    irelevant = models.BooleanField(default=False)

    def __str__(self):
        return f"Event: {self.stock.ticker} - {self.cluster_id}"
