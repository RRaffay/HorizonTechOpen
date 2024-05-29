from django.urls import path
from . import views

urlpatterns = [
    path("analyze/", views.analyze_portfolio, name="analyze_portfolio"),
]
