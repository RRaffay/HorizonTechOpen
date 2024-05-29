from django.http import JsonResponse, HttpResponseServerError
from google.cloud import bigquery
from google.oauth2.service_account import Credentials

# from .utils import (
#     vectorize_columns,
#     cluster_data,
#     silhouette_scores,
#     rank_clusters,
#     extract_important_articles2,
#     to_native_types,
# )
import logging
from django.conf import settings
from .models import GDELTEvent, GDELTCall, GDELTEntry  # Import the new model
from datetime import timedelta
from django.utils import timezone
from user_service.models import Stock

import time
from .utils_embed import (
    preprocess_data_summary,
    create_embedding_df,
    cluster_embeddings,
    rank_clusters,
    extract_important_articles2,
    to_native_types,
)


logger = logging.getLogger(__name__)


credentials = Credentials.from_service_account_file(settings.GOOGLE_CREDENTIALS_PATH)


def add_gdelt_data(df, gdelt_call):
    """
    Store each fetched entry as a GDELTEntry.

    Args:
        df (pandas.DataFrame): The DataFrame containing the fetched entries.

    Returns:
        None
    """
    entries = []
    # Store each fetched entry as a GDELTEntry
    for index, row in df.iterrows():
        entry = GDELTEntry(
            gdelt_call_id=gdelt_call.id,
            date=row["DATE"],
            themes=row["V2Themes"],
            tone=row["V2Tone"],
            locations=row["V2Locations"],
            persons=row["V2Persons"],
            organizations=row["V2Organizations"],
            document_identifier=row["DocumentIdentifier"],
            all_names=row["AllNames"],
            amounts=row["Amounts"],
        )
        entries.append(entry)

    GDELTEntry.objects.bulk_create(entries)


def prepare_gdelt_events(df, top_k=5, n=5, health_cutoff=0):
    """
    Process the to prepare for storing as GDELTEvents.

    Args:
        df (pandas.DataFrame): The DataFrame containing the fetched entries.
        top_k (int): Number of top entities per cluster.
        n (int): Number of top articles per cluster.
        health_cutoff (float): Minimum cluster health to consider.

    Returns:
        dict: A dictionary containing the processed data.
    """
    preprocess_data_summary(df)

    create_embedding_df(df)

    cluster_embeddings(df)

    ranked_clusters = rank_clusters(df, "SummaryEmbedding")

    cluster_info = extract_important_articles2(
        df,
        ranked_clusters=ranked_clusters,
        n=n,
        top_k=top_k,
        health_cutoff=health_cutoff,
    )

    cluster_info = to_native_types(cluster_info)

    return cluster_info


def fetch_gdelt_data3(stock_ticker, interval=7, top_n=10):
    try:
        stock = Stock.objects.get(ticker=stock_ticker)
        client = bigquery.Client(credentials=credentials)
        name_of_company = stock.name.lower()

        sql_query = f"""
        SELECT DATE, V2Themes, V2Tone, V2Locations, V2Persons, V2Organizations, DocumentIdentifier, AllNames, Amounts
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {interval} DAY)
        AND (
        LOWER(V2Organizations) LIKE '%{name_of_company}%')
        """

        df = client.query(sql_query).to_dataframe()

        gdelt_call = GDELTCall.objects.create(
            stock=stock, interval=interval, top_n=top_n
        )

        add_gdelt_data(df, gdelt_call)
        logger.info("Entries saved to the database")

        return JsonResponse("message: Entries added successfully.")

        cluster_info = prepare_gdelt_events(df)

        for cluster, info in cluster_info.items():
            GDELTEvent.objects.create(
                stock=stock,
                cluster_id=cluster,
                top_articles=",".join(info["top_articles"]),
                top_themes=",".join([t[0] for t in info["top_themes"]]),
                top_persons=",".join([p[0] for p in info["top_persons"]]),
                top_orgs=",".join([o[0] for o in info["top_orgs"]]),
                top_locs=",".join([l[0] for l in info["top_locs"]]),
                cluster_health=info["cluster_health"],
                event_window=interval,
                gdelt_call=gdelt_call,
                mean_embedding=info["mean_embedding"],
            )

        print("Events saved to the database")

    except Exception as e:
        print("Error occurred")
        logger.exception(f"An error occurred: {e}")
        raise


# This is to avoid errors. Needs to be removed
def fetch_gdelt_data(stock_ticker, interval=7, top_n=10):
    return JsonResponse("message: Function only a placeholder.")


CONFIG = {
    "health_cutoff": 0,
    "interval": 7,
    "top_n": 10,
    "embedding_column": "SummaryEmbedding",
    "min_cluster_size": 3,
    "metric": "euclidean",
    "n_neighbors_umap": 15,
    "n_components_umap": 2,
    "min_dist_umap": 0.1,
    "umap_metric": "euclidean",
    "number_of_top_articles": 3,
    "number_of_top_entities": 5,
    "size_weigth": 0.5,
    "cohesiveness_weight": 0.5,
}


def fetch_gdelt_data2(stock_ticker, interval=7, top_n=10):
    try:
        stock = Stock.objects.get(ticker=stock_ticker)

        logger.info(f"Gdelt call for {stock_ticker} with interval {interval}")

        start_time = time.time()

        # Initialize BigQuery Client
        client = bigquery.Client(credentials=credentials)
        name_of_company = stock.name.lower()

        sql_query = f"""
        SELECT DATE, V2Themes, V2Tone, V2Locations, V2Persons, V2Organizations, DocumentIdentifier, AllNames, Amounts
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {interval} DAY)
        AND (
        LOWER(V2Organizations) LIKE '%{name_of_company}%')
        """

        df = client.query(sql_query).to_dataframe()

        len(df)

        logger.info(f"Number of entries for Gdelt call for {stock_ticker}: {len(df)}")

        preprocess_data_summary(df)

        create_embedding_df(df)

        cluster_embeddings(
            df,
            embedding_column=CONFIG["embedding_column"],
            min_cluster_size=CONFIG["min_cluster_size"],
            metric=CONFIG["metric"],
            n_neighbors=CONFIG["n_neighbors_umap"],
            n_components=CONFIG["n_components_umap"],
            min_dist=CONFIG["min_dist_umap"],
            umap_metric=CONFIG["umap_metric"],
        )

        ranked_clusters = rank_clusters(
            df,
            embeddings_column=CONFIG["embedding_column"],
            size_weight=CONFIG["size_weigth"],
            cohesiveness_weight=CONFIG["cohesiveness_weight"],
        )

        cluster_info = extract_important_articles2(
            df,
            ranked_clusters=ranked_clusters,
            number_of_top_articles=CONFIG["number_of_top_articles"],
            number_of_top_entities=CONFIG["number_of_top_entities"],
            health_cutoff=CONFIG["health_cutoff"],
        )

        cluster_info = to_native_types(cluster_info)

        gdelt_call = GDELTCall.objects.create(
            stock=stock,
            interval=CONFIG["interval"],
            top_n=top_n,
            config=CONFIG,
        )

        # Save events to the local database
        for cluster, info in cluster_info.items():
            GDELTEvent.objects.create(
                stock=stock,
                cluster_id=cluster,
                top_articles=",".join(info["top_articles"]),
                top_themes=",".join([t[0] for t in info["top_themes"]]),
                top_persons=",".join([p[0] for p in info["top_persons"]]),
                top_orgs=",".join([o[0] for o in info["top_orgs"]]),
                top_locs=",".join([l[0] for l in info["top_locs"]]),
                cluster_health=info["cluster_health"],
                median_date=info["median_date"],
                event_window=interval,
                gdelt_call=gdelt_call,
                mean_embedding=info["mean_embedding"],
            )

        fetch_window = timezone.now() - timedelta(days=interval)

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(
            f"Events for {stock_ticker} for range {fetch_window} saved to the database"
        )
        logger.info(f"Execution time: {execution_time} seconds")

        return

    except Exception as e:
        logger.exception(
            f"An error occurred while trying to fetch events for {stock_ticker}. Error: {e}"
        )
        raise


# def get_gdelt_data(request):
#     keyword = request.GET.get("keyword")
#     interval = int(request.GET.get("interval", 7))
#     top_n = int(request.GET.get("top_n", 10))

#     if not keyword:
#         return JsonResponse({"message": "Keyword is required."})

#     try:
#         cluster_info = fetch_gdelt_data2(keyword, interval, top_n)
#         return JsonResponse(cluster_info)
#     except:
#         return HttpResponseServerError("Internal Server Error")


# def fetch_gdelt_data(stock_ticker, interval=7, top_n=10):
#     try:
#         recent_date = timezone.now() - timedelta(days=interval)

#         stock = Stock.objects.get(ticker=stock_ticker)

#         stored_events = GDELTEvent.objects.filter(
#             stock=stock, created_at__gte=recent_date
#         )

#         if stored_events.exists():
#             # Currently returns the base event object, may be useful to return the serialized version
#             return stored_events

#         # Initialize BigQuery Client
#         client = bigquery.Client(credentials=credentials)
#         name_of_company = stock.name.lower()

#         sql_query = f"""
#         SELECT DATE, V2Themes, V2Tone, V2Locations, V2Persons, V2Organizations, DocumentIdentifier, AllNames, Amounts
#         FROM `gdelt-bq.gdeltv2.gkg_partitioned`
#         WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {interval} DAY)
#         AND (
#         LOWER(V2Organizations) LIKE '%{name_of_company}%')
#         """

#         # Run Query
#         df = client.query(sql_query).to_dataframe()

#         if df.empty:
#             return JsonResponse({"message": "No data found for the given keyword."})

#         # Update Config dynamically
#         min_cluster_size = max(5, int(len(df) * 0.01))

#         # Data processing
#         columns = ["V2Persons", "V2Themes", "V2Organizations", "V2Locations"]
#         X_all = vectorize_columns(df, columns)
#         df["clusterdb"] = cluster_data(X_all, min_cluster_size=min_cluster_size)
#         silhouette_scores(df, X_all)
#         cluster_ranks = rank_clusters(df)

#         # Sort clusters by their cluster_health and get top 10 (or fewer)
#         sorted_clusters = sorted(
#             cluster_ranks.items(), key=lambda x: x[1], reverse=True
#         )[:top_n]

#         # Only extract info for the top 10 clusters
#         top_cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters]
#         filtered_df = df[df["clusterdb"].isin(top_cluster_ids)]

#         cluster_info = extract_important_articles2(
#             filtered_df, X_all, ranked_clusters=cluster_ranks
#         )

#         cluster_info = to_native_types(cluster_info)

#         gdelt_call = GDELTCall.objects.create(
#             stock=stock, interval=interval, top_n=top_n
#         )

#         print("Call saved to the database")

#         # Save events to the local database
#         for cluster, info in cluster_info.items():
#             GDELTEvent.objects.create(
#                 stock=stock,
#                 cluster_id=cluster,
#                 top_articles=",".join(info["top_articles"]),
#                 top_themes=",".join([t[0] for t in info["top_themes"]]),
#                 top_persons=",".join([p[0] for p in info["top_persons"]]),
#                 top_orgs=",".join([o[0] for o in info["top_orgs"]]),
#                 top_locs=",".join([l[0] for l in info["top_locs"]]),
#                 cluster_health=info["cluster_health"],
#                 event_window=interval,
#                 gdelt_call=gdelt_call,
#             )

#         print("Events saved to the database")
#         # If JSON is returned here, then JSON should be returned in the above function as well
#         return JsonResponse("message: Data fetched successfully.")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise
