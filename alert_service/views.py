from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from user_service.models import Portfolio
from django.forms import inlineformset_factory
from .forms import (
    AlertForm,
    AlertStockForm,
    AreaOfInterestForm,
    BaseAreaOfInterestFormSet,
)
from .models import Alert, AlertStock, AreaOfInterest
from django.views.decorators.http import require_POST
import logging

from django.shortcuts import get_object_or_404

logger = logging.getLogger(__name__)


@login_required
def alert_dashboard(request):
    # Create an alert for the user if one doesn't exist
    alert, created = Alert.objects.get_or_create(user=request.user)

    alert_active = alert.is_active

    # Get the user's portfolio
    portfolio = Portfolio.objects.get(user=request.user)

    # Initialize a dictionary to hold stock names and their areas of interest
    stocks_interests = {}

    # Iterate over all stocks in the user's portfolio
    for stock in portfolio.stocks.all():
        # Create an AlertStock for each stock if one doesn't exist
        alert_stock, created = AlertStock.objects.get_or_create(
            alert=alert, stock=stock
        )

        # Get the associated AreaOfInterest objects for the AlertStock
        areas_of_interest = AreaOfInterest.objects.filter(alert_stock=alert_stock)

        logger.info(f"Areas of interest for {stock.name}: {areas_of_interest}")

        # If no AreaOfInterest objects exist, return a string saying "None exist"
        if not areas_of_interest:
            stocks_interests[alert_stock] = "No areas specified"
        else:
            stocks_interests[alert_stock] = "; ".join(
                area.description for area in areas_of_interest
            )

    # Pass the dictionary to the HTML page
    return render(
        request,
        "alert_service/alert_dashboard.html",
        {"stocks_interests": stocks_interests, "alert_active": alert_active},
    )


@login_required
def configure_alerts(request, alert_stock_id):
    # Get the AlertStock object
    alert_stock = AlertStock.objects.get(id=alert_stock_id)

    # Get the associated AreaOfInterest objects for the AlertStock
    areas_of_interest = AreaOfInterest.objects.filter(alert_stock=alert_stock)

    # If this is a POST request, we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request
        form = AreaOfInterestForm(request.POST)
        # Check whether it's valid
        if form.is_valid():
            # Save the new area of interest to the database
            area_of_interest = form.save(commit=False)
            area_of_interest.alert_stock = alert_stock
            area_of_interest.save()
            # Redirect to the alert dashboard after POST
            return redirect("alert_dashboard")
    else:
        # If a GET (or any other method), create the default form
        form = AreaOfInterestForm()

    return render(
        request,
        "alert_service/alert_config.html",
        {
            "form": form,
            "areas_of_interest": areas_of_interest,
            "alert_stock": alert_stock,
        },
    )


@login_required
@require_POST
def edit_area_of_interest(request, area_id):
    area = get_object_or_404(AreaOfInterest, id=area_id)
    form = AreaOfInterestForm(request.POST, instance=area)
    if form.is_valid():
        form.save()
        return JsonResponse({"status": "success"})
    else:
        return JsonResponse({"status": "error", "errors": form.errors})


@login_required
@require_POST
def delete_area_of_interest(request, area_id):
    area = get_object_or_404(AreaOfInterest, id=area_id)
    area.delete()
    return JsonResponse({"status": "success"})
