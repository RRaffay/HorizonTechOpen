from django.shortcuts import get_object_or_404
from stock_news_service.models import GDELTCall, GDELTEvent
from user_service.models import Portfolio, PortfolioStock, Stock
from .models import ProcessedGDELTData
import logging
import time
from .utils import rank_processed_events, check_event_relevance
from stock_news_service.views import fetch_gdelt_data2
from .services import (
    summarize_events,
    rank_events_alert,
    process_query,
)
from stock_news_service.utils_embed import create_embedding
from django.db.models import F, FloatField, ExpressionWrapper
from datetime import date


logger = logging.getLogger(__name__)


def process_GDELT_documents(gdelt_events):
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


def get_stock(stock_ticker):
    return get_object_or_404(Stock, ticker=stock_ticker)


def get_or_fetch_gdelt_call(stock, interval=1):
    today = date.today()
    gdelt_call = (
        GDELTCall.objects.filter(stock=stock, call_time__date=today)
        .order_by("-call_time")
        .first()
    )
    if not gdelt_call:
        fetch_gdelt_data2(stock.ticker, interval=interval)
        gdelt_call = (
            GDELTCall.objects.filter(stock=stock, call_time__date=today)
            .order_by("-call_time")
            .first()
        )
    return gdelt_call


def process_and_rank_gdelt_events(gdelt_events, user_info, top_n_articles=5):
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
    processed_gdelt_events = process_GDELT_documents(gdelt_events=gdelt_events)

    # user_info = process_query(user_info)
    ranked_gdelt_events = rank_processed_events(
        processed_events=processed_gdelt_events, query=user_info
    )[:top_n_articles]
    return [event[0] for event in ranked_gdelt_events]


def get_gdelt_events(gdelt_call, top_n_by_cluster_health=50):
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

    # Annotate each GDELTEvent with a combined_health_score
    gdelt_events = (
        GDELTEvent.objects.filter(gdelt_call=gdelt_call)
        .annotate(
            combined_health_score=ExpressionWrapper(
                F("cluster_health__0") * weight_size_importance
                + F("cluster_health__1") * weight_cohesiveness
                + (1 - F("cluster_health__2")) * weight_time_variance,
                output_field=FloatField(),
            )
        )
        .order_by("-combined_health_score")[:top_n_by_cluster_health]
    )

    return gdelt_events


def alert_analysis(stock_interest_dict):
    """
    For each stock, fetch GDELT data for the last day, process the events, rank these events.
    Finally, generate a combined summary of all the rankings.

    Args:
        stock_interest_dict: A dictionary where keys are stock names and values are areas_of_interest.

    Returns:
        A dictionary with the combined summary of all the rankings for each stock.
    """
    combined_summary = {}

    for stock_name, areas_of_interest in stock_interest_dict.items():
        # Get the stock object
        stock = get_object_or_404(Stock, name=stock_name)

        # Fetch GDELT data for the last day
        gdelt_call = get_or_fetch_gdelt_call(stock, interval=1)

        combined_summary[stock_name] = ""

        for area_of_interest in areas_of_interest:
            # Process the area of interest
            processed_area_of_interest = process_query(area_of_interest)

            # Get GDELT events
            gdelt_events = get_gdelt_events(gdelt_call, top_n_by_cluster_health=30)

            # Process and rank GDELT events
            processed_gdelt_events = process_and_rank_gdelt_events(
                gdelt_events, processed_area_of_interest, top_n_articles=30
            )

            # Generate a string representation of the processed GDELT data
            processed_gdelt_data = [
                {
                    "summary": p_event.processed_data,
                    "links": p_event.event.top_articles.split(","),
                    "event_health": p_event.event.cluster_health,
                }
                for p_event in processed_gdelt_events
            ]

            processed_string = generate_processed_gdelt_string(processed_gdelt_data)

            # Rank events
            rank_events2_response = rank_events2(
                company_name=stock.name,
                events=processed_string,
                context=area_of_interest,
            )

            # Add the combined summary to the dictionary
            combined_summary[stock_name] += rank_events2_response + "\n"

        # TODO: Add a function that combines the summaries for each stock

    logger.info(combined_summary)

    return combined_summary
