from django.http import JsonResponse, HttpResponseBadRequest
from .services import query_filings, extract_key_items
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import SECFiling


@login_required
def fetch_sec_filings_for_stock(request, tickers):
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method")

    form_type = request.POST.get(
        "form_type", "10-K"
    )  # Default to "10-K" if not specified
    if form_type not in ["10-K", "10-Q", "8-K"]:
        return HttpResponseBadRequest("Invalid form type provided")

    sections = request.POST.get("sections", "").split(",")
    sections = [s.strip() for s in sections if s.strip()]

    if not tickers:
        return HttpResponseBadRequest("Invalid tickers provided")

    tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

    # Modify this line to pass form_type and sections to query_filings
    metadata = query_filings(tickers, form_type)

    # This has been added for testing, this should be changed to be more dynamic
    sections = ["1", "1A", "7"]

    # Modify this line to pass sections to extract_key_items
    result = extract_key_items(tickers, sections)
    return JsonResponse(result)


@login_required
def filings_dashboard(request, ticker):
    ticker = request.GET.get("ticker", "")

    if ticker:
        filings = SECFiling.objects.filter(ticker=ticker).order_by("-filedAt")
    else:
        filings = SECFiling.objects.all().order_by("-filedAt")[
            :10
        ]  # Display the latest 10 filings by default

    return render(request, "dashboard.html", {"filings": filings, "ticker": ticker})
