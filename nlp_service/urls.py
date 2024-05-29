from django.urls import path
from . import views

urlpatterns = [
    path("view_news/<str:stock_ticker>/", views.view_news, name="view_news"),
    path(
        "view_processed_sec_filing/<str:stock_ticker>/",
        views.view_processed_sec_filing,
        name="view_processed_sec_filing",
    ),
    path("change_stock_info", views.change_stock_info, name="change_stock_info"),
    path("get_news_answer", views.get_news_answer, name="get_news_answer"),
]
