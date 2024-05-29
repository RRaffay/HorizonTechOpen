from django.urls import path
from . import views


urlpatterns = [
    path("alert_dashboard", views.alert_dashboard, name="alert_dashboard"),
    path(
        "configure-alerts/<int:alert_stock_id>",
        views.configure_alerts,
        name="configure-alerts",
    ),
    path(
        "edit_area_of_interest/<int:area_id>/",
        views.edit_area_of_interest,
        name="edit_area_of_interest",
    ),
    path(
        "delete_area_of_interest/<int:area_id>/",
        views.delete_area_of_interest,
        name="delete_area_of_interest",
    ),
]
