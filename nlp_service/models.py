from django.db import models
from sec_filings_service.models import SECFiling
from stock_news_service.models import GDELTEvent, GDELTCall
from django.core.serializers.json import DjangoJSONEncoder
import random


class ProcessedSECData(models.Model):
    filing = models.ForeignKey(SECFiling, on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    processed_data = models.JSONField()
    # Foreign Key to Stock
    ticker = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.filing.ticker} - Processed at {self.generated_at}"


def default_float_list():
    # Should be set to default to embedding size of text-ada-02
    return [-10 for _ in range(1536)]


class ProcessedGDELTData(models.Model):
    event = models.ForeignKey(GDELTEvent, on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    processed_data = models.JSONField()
    summary_embedding = models.JSONField(
        default=default_float_list, encoder=DjangoJSONEncoder
    )

    # Foreign Key to Stock
    def __str__(self):
        return f"{self.event} - Processed at {self.generated_at}"
