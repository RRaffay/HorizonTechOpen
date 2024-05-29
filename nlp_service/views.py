from django.http import JsonResponse
from nlp_service.services import (
    summarize_events,
    rank_events2,
    process_query,
)
from nlp_service.models import ProcessedSECData, ProcessedGDELTData
from django.contrib.auth.decorators import login_required
from user_service.models import Stock
from stock_news_service.models import GDELTEvent, GDELTCall
from stock_news_service.views import fetch_gdelt_data2
import json as json
from django.shortcuts import render, redirect
from .utils import rank_processed_events, check_event_relevance
from user_service.models import Portfolio, PortfolioStock
from django.shortcuts import get_object_or_404
from stock_news_service.utils_embed import create_embedding
import datetime
from django.views.decorators.csrf import csrf_exempt
from django.db.models import F, FloatField, ExpressionWrapper
from retriever_processing.views import process_sec_document
import time
from django_ratelimit.decorators import ratelimit
from django.core.cache import cache
from django.db.models.functions import Cast

import logging

logger = logging.getLogger(__name__)


def process_GDELT_documents(request, gdelt_events):
    """
    Process GDELT documents and store the processed data in the database if the event is relevant to the stock.

    Args:
        gdelt_events: List of GDELTEvent objects.

    Returns:
        List of ProcessedGDELTData objects.
    """
    i = 1
    # Checking if processed event exists in the db

    start_time = time.time()

    total_events = len(gdelt_events)
    logger.info(f"Processing {total_events} GDELT Events")
    for event in gdelt_events:
        logger.info("Processing event: %s", i)
        # Check relevance
        gdelt_event = GDELTEvent.objects.filter(id=event.id).first()
        if gdelt_event.irelevant:
            i += 1
            continue

        processed_gdelt_data = ProcessedGDELTData.objects.filter(event=event).first()
        if not processed_gdelt_data:
            logger.info("Processing GDELT Event started")
            # Process the event

            stock_name = event.stock.name  # Get the stock name from the event

            processed_data = summarize_events(gdelt_event=event, stock_name=stock_name)
            summary_embedding = create_embedding(processed_data)

            # Check if the event is relevant to the stock
            is_relevant = check_event_relevance(processed_data)

            # Store the processed data in the database if the event is relevant
            if is_relevant:
                processed_gdelt_data = ProcessedGDELTData(
                    event=event,
                    processed_data=processed_data,
                    summary_embedding=summary_embedding,
                )
                # Save the processed data
                processed_gdelt_data.save()
                logger.info("Processing of GDELT Event is complete")
            else:
                logger.info("Event is not relevant")
                # Print details about the event
                logger.info("Event ID: %s", event.id)  # Get the event ID
                # Get the GDELTEvent object set irelevant to True
                # Set irelevant attribute to True
                event.irelevant = True
                event.save()
        i += 1

    # Get the corresponding processed GDELT data for each event
    processed_gdelt_data_list = []
    for event in gdelt_events:
        processed_gdelt_data = ProcessedGDELTData.objects.filter(event=event).first()
        if processed_gdelt_data:
            processed_gdelt_data_list.append(processed_gdelt_data)

    logger.info("Event Processing Complete")

    end_time = time.time()

    execution_time = end_time - start_time

    logger.info(f"Processing {total_events} GDELT Events took {execution_time} seconds")

    return processed_gdelt_data_list


def sec_processed_string_func(processed_data):
    ret = ""
    for query_response in processed_data.processed_data.values():
        ret += (
            "Question:\n"
            + query_response["query"]
            + "\n Answer:\n"
            + query_response["result"]
            + "\n"
        )

    return ret


def gdelt_event_string_func(processed_gdelt_data):
    """
    Generate a string representation of the processed GDELT data.

    Args:
        processed_gdelt_data: List of ProcessedGDELTData objects.

    Returns:
        String representation of the processed GDELT data.
    """
    ret = ""
    for event in processed_gdelt_data:
        for event_summary in event.values():
            ret += "\n Event Summary:\n" + event_summary + "\n\n"

    return ret


def generate_processed_gdelt_string(processed_gdelt_data):
    """
    Generate a string representation of the processed GDELT data.

    Args:
        processed_gdelt_data: List of processed GDELT data.

    Returns:
        String representation of the processed GDELT data.
    """
    processed_gdelt_string = ""
    for p_event in processed_gdelt_data:
        processed_gdelt_string += f""" 
        Summary: "{p_event["summary"]}" 
"""
    return processed_gdelt_string


@login_required
def get_user_portfolio(request):
    return get_object_or_404(Portfolio, user=request.user)


def get_stock(stock_ticker):
    return get_object_or_404(Stock, ticker=stock_ticker)


def get_portfolio_stock(portfolio, stock):
    return PortfolioStock.objects.get(portfolio=portfolio, stock=stock)


def get_or_fetch_gdelt_call(stock, interval=7):
    gdelt_call = GDELTCall.objects.filter(stock=stock).order_by("-call_time").first()
    if not gdelt_call:
        fetch_gdelt_data2(stock.ticker, interval=interval)
        gdelt_call = (
            GDELTCall.objects.filter(stock=stock).order_by("-call_time").first()
        )
    return gdelt_call


def process_and_rank_gdelt_events(request, gdelt_events, user_info, top_n_articles=5):
    """
    Process and rank GDELT events.

    Args:
        request: The HTTP request object.
        gdelt_events: List of GDELTEvent objects.
        user_info: User defined context of interest
        top_n_articles: Number of top articles to return.

    Returns:
        List of processed GDELT events.
    """
    processed_gdelt_events = process_GDELT_documents(request, gdelt_events=gdelt_events)

    # user_info = process_query(user_info)
    ranked_gdelt_events = rank_processed_events(
        processed_events=processed_gdelt_events, query=user_info
    )[:top_n_articles]
    return [event[0] for event in ranked_gdelt_events]


def get_gdelt_events(gdelt_call, top_n_by_cluster_health=50, date_range=None):
    """
    Get GDELT events based on the cluster health scores and date range.

    Args:
        gdelt_call: GDELTCall object.
        top_n_by_cluster_health: Number of events to retrieve based on cluster health scores.
        date_range: Tuple of start and end dates.

    Returns:
        QuerySet of GDELTEvent objects.
    """
    # Define weights for each component of cluster_health
    weight_size_importance = 0.5
    weight_cohesiveness = 0.3
    # Since a higher time_variance indicates poorer health, its weight is negative.
    weight_time_variance = -0.2
    if date_range == None:
        # Annotate each GDELTEvent with a combined_health_score
        gdelt_events = (
            GDELTEvent.objects.filter(gdelt_call=gdelt_call)
            .annotate(
                combined_health_score=ExpressionWrapper(
                    Cast("cluster_health__0", FloatField()) * weight_size_importance
                    + Cast("cluster_health__1", FloatField()) * weight_cohesiveness
                    + (1 - Cast("cluster_health__2", FloatField()))
                    * weight_time_variance,
                    output_field=FloatField(),
                )
            )
            .order_by("-combined_health_score")[:top_n_by_cluster_health]
        )

        return gdelt_events
    else:
        start_date, end_date = date_range
        gdelt_events = (
            GDELTEvent.objects.filter(
                gdelt_call=gdelt_call, median_date__range=[start_date, end_date]
            )
            .annotate(
                combined_health_score=ExpressionWrapper(
                    Cast("cluster_health__0", FloatField()) * weight_size_importance
                    + Cast("cluster_health__1", FloatField()) * weight_cohesiveness
                    + (1 - Cast("cluster_health__2", FloatField()))
                    * weight_time_variance,
                    output_field=FloatField(),
                )
            )
            .order_by("-combined_health_score")[:top_n_by_cluster_health]
        )

        return gdelt_events


@login_required
@ratelimit(key="user", rate="10/m", method=ratelimit.ALL, block=True)
@ratelimit(key="user", rate="1000/d", method=ratelimit.ALL, block=True)
def view_news(request, stock_ticker):
    """
    View news related to a specific stock.

    Args:
        request: The HTTP request object.
        stock_ticker: Ticker symbol of the stock.

    Returns:
        HTTP response with the rendered news view.
    """
    portfolio = get_user_portfolio(request)
    stock = get_stock(stock_ticker)
    portfolio_stock = get_portfolio_stock(portfolio, stock)

    user_info_display = portfolio_stock.area_of_interest

    user_info = portfolio_stock.processed_area_of_interest

    gdelt_call = get_or_fetch_gdelt_call(stock, interval=7)

    max_date = gdelt_call.call_time.date()
    min_date = (
        gdelt_call.call_time - datetime.timedelta(days=gdelt_call.interval)
    ).date()

    date_range = (
        request.POST.get("startDate", None),
        request.POST.get("endDate", None),
    )

    if date_range == (None, None):
        date_range = None

    if date_range:
        start_date = date_range[0]
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

        end_date = date_range[1]
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        logger.info("Custom Date range: %s", date_range)

        if start_date < min_date or end_date > max_date:
            start_date = min_date
            end_date = max_date

        events_range = str(date_range[0]) + " to " + str(date_range[1])
    else:
        events_range = (
            str(
                (
                    gdelt_call.call_time - datetime.timedelta(days=gdelt_call.interval)
                ).date()
            )
            + " to "
            + str(gdelt_call.call_time.date())
        )

    gdelt_events = get_gdelt_events(
        gdelt_call, top_n_by_cluster_health=30, date_range=date_range
    )

    # Get the number of events to analyze from the POST data
    num_events = int(request.POST.get("numEvents", 30))

    processed_gdelt_events = process_and_rank_gdelt_events(
        request, gdelt_events, user_info, top_n_articles=num_events
    )

    processed_gdelt_data = [
        {
            "summary": p_event.processed_data,
            "links": p_event.event.top_articles.split(
                ","
            ),  # Make sure this is a list of URLs
            "event_health": p_event.event.cluster_health,  # Add the event health
        }
        for p_event in processed_gdelt_events
    ]

    processed_string = generate_processed_gdelt_string(processed_gdelt_data)

    # Generate a unique cache key based on the function's inputs
    cache_key = f"rank_events2_{stock.name}_{processed_string}_{user_info}"
    cache_timeout = 60 * 60  # for example, cache for 1 hour

    # Try to get the cached result
    rank_events2_response = cache.get(cache_key)

    # If it's not cached, compute the result and cache it
    if rank_events2_response is None:
        logger.info("Ranking events using rank_events2. Cache miss")
        rank_events2_response = rank_events2(
            company_name=stock.name, events=processed_string, context=user_info
        )
        # Set the result in the cache
        cache.set(cache_key, rank_events2_response, cache_timeout)

    else:
        logger.info("Using cached ranking")

    return render(
        request,
        "nlp_service/view_news.html",
        {
            "ticker": stock_ticker,
            "processed_data_list": processed_gdelt_data,
            "user_info": user_info_display,
            "llm_ranking": rank_events2_response,
            "events_range": events_range,
            "min_date": min_date,
            "max_date": max_date,
        },
    )


from .chatbot_utils import ChatbotService


@login_required
def get_news_answer(request):
    """
    Get the answer to a news-related question.

    Args:
        request: The HTTP request object.

    Returns:
        JSON response with the answer.
    """
    if request.method == "POST":
        user = request.user
        if not user.profile.is_paying_customer:
            return JsonResponse(
                {"answer": "Please subscribe to  Pro plan to use this feature"}
            )

        data = json.loads(request.body)
        question = data.get("question", "What is the revenue of Apple?")
        chat_history_tuples = data.get("chat_history", [("Hello", "Hi")])
        analysis = data.get("analysis", "No Analysis")
        analysis_context = data.get("analysis_context", " ")

        chatbot_service = ChatbotService(
            analysis=analysis, analysis_context=analysis_context
        )
        result = chatbot_service.get_response(
            question=question, chat_history_tuples=chat_history_tuples
        )

        return JsonResponse({"answer": result})
    else:
        return JsonResponse({"Error": "Invalid request"}, status=400)


@login_required
@ratelimit(key="user", rate="5/m", method=ratelimit.ALL, block=True)
@ratelimit(key="user", rate="100/d", method=ratelimit.ALL, block=True)
def change_stock_info(request):
    """
    Change the stock information.

    Args:
        request: The HTTP request object.

    Returns:
        Redirect to the view_news page.
    """
    if request.method == "POST":
        portfolio = get_object_or_404(Portfolio, user=request.user)

        # Retrieve the specific stock
        stock = get_object_or_404(Stock, ticker=request.POST["ticker"])

        portfolio_stock = PortfolioStock.objects.get(portfolio=portfolio, stock=stock)

        # Update the user info
        portfolio_stock.area_of_interest = request.POST["info"]
        portfolio_stock.processed_area_of_interest = process_query(
            portfolio_stock.area_of_interest
        )
        logger.info(portfolio_stock.area_of_interest)
        portfolio_stock.save()

        return redirect("view_news", stock_ticker=request.POST["ticker"])


@login_required
def view_processed_sec_filing(request, stock_ticker):
    """
    View processed SEC filing for a specific stock.

    Args:
        request: The HTTP request object.
        stock_ticker: Ticker symbol of the stock.

    Returns:
        HTTP response with the rendered processed SEC filing view.
    """
    processed_sec_filing = ProcessedSECData.objects.filter(ticker=stock_ticker).first()

    if processed_sec_filing:
        return render(
            request,
            "nlp_service/view_processed_sec.html",
            {"processed_sec_filing": processed_sec_filing},
        )
    else:
        processed_doc, filing = process_sec_document(request, stock_ticker)
        new_processed_sec_filing = ProcessedSECData.objects.create(
            ticker=stock_ticker, filing=filing, processed_data=processed_doc
        )
        new_processed_sec_filing.save()

        processed_sec_filing = ProcessedSECData.objects.filter(
            ticker=stock_ticker
        ).first()
        return render(
            request,
            "nlp_service/view_processed_sec.html",
            {"processed_sec_filing": processed_sec_filing},
        )
