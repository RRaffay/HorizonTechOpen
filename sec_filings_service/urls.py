from django.urls import path
from . import views

urlpatterns = [
    # ... other URL patterns ...
    path(
        "fetch-sec-filings/<str:tickers>/",
        views.fetch_sec_filings_for_stock,
        name="fetch_sec_filings",
    ),
]
