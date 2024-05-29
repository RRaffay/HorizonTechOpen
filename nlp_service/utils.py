from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from stock_news_service.utils_embed import create_embedding
from nlp_service.models import ProcessedGDELTData


def create_cluster_embeddings(event_cluster):
    """
    Create an embedding for a cluster.

    Args:
        events(dict): dictionary with events details (see models.py)

    Returns:
        cluster_embedding (list): embeddings for the meta-data of the cluster
    """

    meta_data_string = "This is an event where "

    for key, value in event_cluster.items():
        meta_data_string += key + ": " + str(value) + " "

    cluster_embedding = create_embedding(meta_data_string)
    return cluster_embedding


def check_event_relevance(event_summary):
    """
    Check if the event is relevant to the stock.

    Args:
        event_summary (str): summary of the event
        stock_name (str): name of the stock

    Returns:
        is_relevant (bool): True if the event is relevant to the stock, False otherwise

    """

    irrelevance_embedding = create_embedding("Irrelevant Article")
    irrelevance_embedding = (np.array(irrelevance_embedding)).reshape(1, -1)

    event_embedding = create_embedding(event_summary)
    event_embedding = (np.array(event_embedding)).reshape(1, -1)

    similarity = cosine_similarity(event_embedding, irrelevance_embedding)
    print(similarity)
    if similarity > 0.8:
        return False
    else:
        return True


def rank_events_description(events, description):
    """
    Rank the events based on cosine similarity for description.

    Args:
        events (list): list of tuples with (event_id, {dict with event details (see models.py)})
        query (str): The query to search for.

    Returns:
        cluster_rank (list): sorted clusters by relevance to query.

    """
    query_embedding = np.array(create_embedding(description))
    query_embedding_2d = query_embedding.reshape(1, -1)
    cluster_rank = []
    for event in events:
        event_embedding = np.array(create_cluster_embeddings(event[1]))
        event_embedding_2d = event_embedding.reshape(1, -1)
        similarity = cosine_similarity(event_embedding_2d, query_embedding_2d)
        event_sim = (event[0], similarity)
        cluster_rank.append(event_sim)

    cluster_rank = sorted(cluster_rank, key=lambda x: x[1], reverse=True)

    return cluster_rank


def rank_processed_events(processed_events, query):
    """
    Rank the events based on cosine similarity for description.

    Args:
        processed_events (ProcessedGDELTData):  (see models.py) )
        query (str): The query to search for.

    Returns:
        cluster_rank (list): sorted clusters by relevance to query.

    """
    query_embedding = np.array(create_embedding(query))
    query_embedding_2d = query_embedding.reshape(1, -1)
    cluster_rank = []
    for p_event in processed_events:
        if p_event.summary_embedding[0] == -10:
            p_event.summary_embedding = create_embedding(p_event.processed_data)
            p_event.save()
            event_embedding = np.array(p_event.summary_embedding)
        else:
            event_embedding = np.array(p_event.summary_embedding)
        event_embedding_2d = event_embedding.reshape(1, -1)
        similarity = cosine_similarity(event_embedding_2d, query_embedding_2d)
        event_sim = (p_event, similarity)
        cluster_rank.append(event_sim)

    cluster_rank = sorted(cluster_rank, key=lambda x: x[1], reverse=True)

    return cluster_rank
