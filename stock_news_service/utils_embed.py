# Imports for functions
import pandas as pd
import re
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


import umap
from sentence_transformers import SentenceTransformer
import datetime
import time


import logging

logger = logging.getLogger(__name__)


client_open_ai = OpenAI()


def get_cluster_median_date(df, cluster_id):
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S")

    cluster_df = df[df["Cluster"] == cluster_id]
    median_date = cluster_df["DATE"].median().to_pydatetime()

    return median_date


def date_converter(date_str):
    date_str = str(date_str)
    date = datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return date.strftime("%dth %B %Y")


def preprocess(entry, column_type):
    """
    Preprocesses the given entry based on the specified column type.

    Args:
        entry (str): The entry to preprocess.
        column_type (str): The type of column.

    Returns:
        str: The preprocessed entry.
    """

    if not isinstance(entry, str):
        return ""
    mentions = entry.split(";")
    if column_type == "location":
        names = [mention.split("#")[2] for mention in mentions]
    else:
        names = [mention.split(",")[0].replace(" ", "_") for mention in mentions]
    return ", ".join(set(names))


def preprocess_data_summary(df):
    """
    Combine the processed columns into a summary column that is used to create the embeddings

    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """

    required_columns = ["V2Persons", "V2Organizations", "V2Locations", "V2Themes"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Applying the preprocess function to each column
    df["ProcessedPersons"] = df["V2Persons"].apply(
        lambda x: re.sub(r"[\d_]", " ", preprocess(x, "other"))
    )
    df["ProcessedOrganizations"] = df["V2Organizations"].apply(
        lambda x: re.sub(r"[\d_]", " ", preprocess(x, "other"))
    )
    df["ProcessedLocations"] = df["V2Locations"].apply(
        lambda x: re.sub(r"[\d_]", " ", preprocess(x, "location"))
    )
    df["ProcessedThemes"] = df["V2Themes"].apply(
        lambda x: re.sub(r"[\d_]", " ", preprocess(x, "other").lower())
    )

    df["ProcessedDate"] = df["DATE"].apply(date_converter)

    # Constructing the new column with the desired format
    df["Summary"] = df.apply(
        lambda row: f"This article talks about these people ({row['ProcessedPersons']}), "
        f"these organizations ({row['ProcessedOrganizations']}), "
        f"these locations ({row['ProcessedLocations']}). It took place on {row['ProcessedDate']} and includes "
        f"these themes ({row['ProcessedThemes']}).",
        axis=1,
    )

    # Now digits are removed from all columns except 'ProcessedDate'

    return df


def create_openai_embedding(text):
    max_retries = 3  # Set the maximum number of retries
    wait_times = [5, 10]  # Wait times before each retry (except the last one)

    for attempt in range(max_retries):
        try:
            response = client_open_ai.embeddings.create(
                input=text, model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:  # Check if we have more retries left
                # Print an error message with the attempt count
                logger.info(f"An error occurred on attempt {attempt + 1}: {e}")
                time.sleep(
                    wait_times[attempt]
                )  # Wait for the specified time before the next retry

            else:
                # If all retries have been used, return None
                logger.exception(f"All retries failed. An error occurred: {e}")
                return None


# Function to create embeddings for the entire dataframe using Sentence-Transformers
def create_sentence_transformers_embeddings(df, column_name):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    sentences = df[column_name].tolist()
    embeddings = model.encode(sentences)
    return embeddings


def create_embedding_df(df, column_name="Summary"):
    # Attempt to create embeddings using OpenAI
    df["SummaryEmbedding"] = df[column_name].apply(create_openai_embedding)

    # Check if any of the embeddings are None (which indicates a failure)
    if df["SummaryEmbedding"].isnull().any():
        print(
            "OpenAI embeddings failed, using Sentence-Transformers for all embeddings."
        )
        df["SummaryEmbedding"] = create_sentence_transformers_embeddings(
            df, column_name
        )

    return df


# Function to create an embedding for a given text using Sentence-Transformers
def create_sentence_transformers_embedding(text):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embedding = model.encode([text])[0]
    return embedding


def create_embedding(text):
    try:
        return create_openai_embedding(text)
    except Exception as e:
        logger.exception(f"OpenAI embeddings failed. An error occurred: {e}")
        return create_sentence_transformers_embedding(text)


# def cluster_embeddings(
#     df, embedding_column="SummaryEmbedding", min_cluster_size=3, metric="euclidean"
# ):
#     """
#     Apply HDBSCAN clustering to embeddings in a DataFrame and add cluster labels to the DataFrame.

#     Parameters:
#     df (pd.DataFrame): DataFrame containing the embeddings.
#     embedding_column (str): Column name of the DataFrame where embeddings are stored.
#     min_cluster_size (int): The minimum size of clusters.
#     metric (str): The metric to use when calculating distance between instances in a feature array.

#     Returns:
#     pd.DataFrame: DataFrame with an additional column 'Cluster' containing cluster labels.
#     """

#     # Check if the embedding column exists in the DataFrame
#     if embedding_column not in df.columns:
#         raise ValueError(
#             f"The DataFrame does not contain the column '{embedding_column}'."
#         )

#     # Convert the list of embeddings to a NumPy array
#     embeddings_array = np.array(df[embedding_column].tolist())

#     # Perform HDBSCAN clustering
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
#     cluster_labels = clusterer.fit_predict(embeddings_array)

#     # Add the cluster labels to the DataFrame
#     df["Cluster"] = cluster_labels

#     return df


def cluster_embeddings(
    df,
    embedding_column="SummaryEmbedding",
    min_cluster_size=3,
    metric="euclidean",
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    umap_metric="euclidean",
):
    """
    Apply UMAP dimensionality reduction followed by HDBSCAN clustering to embeddings in a DataFrame and
    add cluster labels to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the embeddings.
    embedding_column (str): Column name of the DataFrame where embeddings are stored.
    min_cluster_size (int): The minimum size of clusters for HDBSCAN.
    metric (str): The metric to use when calculating distance between instances for HDBSCAN.
    n_neighbors (int): The number of neighbors to consider for each point for UMAP.
    n_components (int): The number of dimensions to reduce to for UMAP.
    min_dist (float): The effective minimum distance between embedded points for UMAP.
    umap_metric (str): The metric to use when calculating distance between instances for UMAP.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'Cluster' containing cluster labels.
    """

    # Check if the embedding column exists in the DataFrame
    if embedding_column not in df.columns:
        raise ValueError(
            f"The DataFrame does not contain the column '{embedding_column}'."
        )

    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(df[embedding_column].tolist())

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=umap_metric,
    )
    reduced_embeddings = reducer.fit_transform(embeddings_array)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Add the cluster labels to the DataFrame
    df["Cluster"] = cluster_labels

    return df


def extract_diverse_top_articles(
    cluster_embeddings, centroid, number_of_top_articles=5, similarity_threshold=0.95
):
    """
    Extracts top n articles that are not too similar to each other.

    Args:
        cluster_embeddings (numpy.ndarray): Array of embeddings of articles within a cluster.
        centroid (numpy.ndarray): Centroid of the cluster.
        number_of_top_articles (int): Number of top articles to select.
        similarity_threshold (float): Threshold for similarity between selected articles.

    Returns:
        list: Indices of the top n diverse articles.
    """

    # Rank articles based on their distance to the centroid
    distances_to_centroid = cosine_similarity(
        cluster_embeddings, centroid.reshape(1, -1)
    )
    sorted_indices = np.argsort(distances_to_centroid.flatten())[::-1]

    selected_indices = [sorted_indices[0]]  # start with the closest article
    for idx in sorted_indices[1:]:
        is_too_similar = False
        for selected_idx in selected_indices:
            if (
                cosine_similarity(
                    cluster_embeddings[idx].reshape(1, -1),
                    cluster_embeddings[selected_idx].reshape(1, -1),
                )[0][0]
                > similarity_threshold
            ):
                is_too_similar = True
                break

        if not is_too_similar:
            selected_indices.append(idx)

        if len(selected_indices) == number_of_top_articles:
            break

    return selected_indices


def extract_top_entities(column_name, df, number_of_top_entities=5):
    """
    Extracts the top n entities by frequency from a column for each cluster.

    Args:
        column_name (str): The dataframe column to analyze.
        df (pandas.DataFrame): The clustered dataframe.
        number_of_top_entities (int): Number of top entities to return per cluster.

    Returns:
        dict: Mapping of cluster IDs to list of top (entity, count) tuples.
    """

    clusters = {}
    for i in df["Cluster"].unique():
        cluster_docs = df[df["Cluster"] == i][column_name]
        entity_count = {}

        for doc in cluster_docs:
            if isinstance(doc, str):
                for entity in doc.split(";"):
                    if column_name == "V2Locations":
                        split_entity = entity.split("#")
                        if len(split_entity) > 2:
                            name = split_entity[2]
                        else:
                            continue
                    else:
                        name = entity.split(",")[0]

                    if name != "":
                        entity_count[name] = entity_count.get(name, 0) + 1

        clusters[i] = sorted(entity_count.items(), key=lambda x: x[1], reverse=True)[
            :number_of_top_entities
        ]
    return clusters


def extract_important_articles2(
    df,
    number_of_top_articles=5,
    ranked_clusters=None,
    number_of_top_entities=5,
    health_cutoff=0,
):
    """
    Extracts the top articles and entities for each cluster.

    Args:
        df (pandas.DataFrame): Clustered dataframe.
        number_of_top_articles (int): Number of top articles per cluster.
        ranked_clusters (dict): Cluster rankings, if available.
        number_of_top_entities (int): Number of top entities per cluster.
        health_cutoff (float): Minimum cluster health to consider.

    Returns:
       dict: Mapping of cluster IDs to extracted info.
    """
    unique_clusters = df["Cluster"].unique()
    cluster_info = {}

    for cluster in unique_clusters:
        # Skip cluster with ID -1
        if cluster == -1:
            continue

        cluster_health = (
            ranked_clusters.get(cluster, "Unknown") if ranked_clusters else "Unknown"
        )

        # Only consider clusters with a cluster_health greater than 0
        if isinstance(cluster_health, (int, float)) and cluster_health <= health_cutoff:
            continue

        # Select articles in the cluster
        cluster_articles = df[df["Cluster"] == cluster]

        # If there are no articles in the cluster, continue to the next iteration
        if cluster_articles.shape[0] == 0:
            continue

        # Extract meta-data for the top articles (assuming you have a way to select them)
        top_articles_metadata = cluster_articles.head(number_of_top_articles)

        # Extracting the embeddings for the article
        ######
        # Extract all embeddings in the cluster
        embeddings_in_cluster = np.array(
            df[df["Cluster"] == cluster]["SummaryEmbedding"].tolist()
        )

        # Calculate the centroid (mean) of the embeddings in the cluster
        centroid = np.mean(embeddings_in_cluster, axis=0)
        # Initialize UMAP with 100 components
        reducer = umap.UMAP(n_components=100)

        # Fit and transform the data to the UMAP model
        # Note: UMAP expects 2D array, so we need to add an extra dimension to centroid
        centroid = np.expand_dims(centroid, axis=0)
        reduced_centroid = reducer.fit_transform(centroid)

        # Since fit_transform returns 2D array, we take the first (and only) element
        reduced_centroid = reduced_centroid[0]
        centroid = reduced_centroid.tolist()

        #####
        # Get the median date of the cluster
        median_date = get_cluster_median_date(df, cluster)

        # Extract the most frequently occurring entities
        top_themes = extract_top_entities(
            "V2Themes", cluster_articles, number_of_top_entities
        )
        top_persons = extract_top_entities(
            "V2Persons", cluster_articles, number_of_top_entities
        )
        top_orgs = extract_top_entities(
            "V2Organizations", cluster_articles, number_of_top_entities
        )
        top_locs = extract_top_entities(
            "V2Locations", cluster_articles, number_of_top_entities
        )

        cluster_info[cluster] = {
            "top_articles": top_articles_metadata["DocumentIdentifier"].tolist(),
            "top_themes": top_themes[cluster],
            "top_persons": top_persons[cluster],
            "top_orgs": top_orgs[cluster],
            "top_locs": top_locs[cluster],
            "cluster_health": cluster_health,
            "mean_embedding": centroid,
            "median_date": median_date,
        }

    return cluster_info


def get_cluster_date_variance(cluster_dates):
    mean = cluster_dates.mean()
    seconds = (cluster_dates - mean).dt.total_seconds()
    var = (seconds**2).sum() / len(seconds)
    return var


def calculate_cluster_cohesiveness(cluster_embeddings):
    """
    Calculate the cohesiveness of a cluster based on the average pairwise cosine similarity.
    """
    if len(cluster_embeddings) <= 1:
        return 0
    similarities = cosine_similarity(cluster_embeddings)
    # Exclude the diagonal (self-similarity) from the calculation
    np.fill_diagonal(similarities, 0)
    average_similarity = np.mean(similarities)
    return average_similarity


def rank_clusters(
    df, embeddings_column, size_weight=0.5, cohesiveness_weight=0.5, variance_weight=0.5
):
    """
    Rank clusters based on the average closeness of nodes, the number of nodes in the cluster,
    and the date variance within the cluster.

    Args:
        df (pandas.DataFrame): The dataframe with clustering results.
        embeddings_column (str): The name of the column containing embeddings.
        size_weight (float): The weight for cluster size.
        cohesiveness_weight (float): The weight for cluster cohesiveness.
        variance_weight (float): The weight for cluster date variance.

    Returns:
        dict: Dictionary mapping cluster IDs to their importance scores.
    """
    ranked_clusters = {}
    unique_clusters = df["Cluster"].unique()
    cluster_variances = {}

    # Calculate cluster sizes and cohesiveness
    cluster_sizes = df["Cluster"].value_counts()
    cluster_cohesiveness = {}

    # First calculate variance for each cluster
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise cluster
            continue
        cluster_df = df[df["Cluster"] == cluster]
        dates = pd.to_datetime(cluster_df["DATE"], format="%Y%m%d%H%M%S")
        cluster_variances[cluster] = get_cluster_date_variance(dates)

    # Normalize the variances
    max_variance = max(cluster_variances.values())
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise cluster
            continue
        cluster_variances[cluster] /= max_variance

    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise cluster
            continue
        cluster_embeddings = np.array(
            df[df["Cluster"] == cluster][embeddings_column].tolist()
        )
        cluster_cohesiveness[cluster] = calculate_cluster_cohesiveness(
            cluster_embeddings
        )

    # Normalize the sizes and cohesiveness
    max_size = max(cluster_sizes)
    max_cohesiveness = max(cluster_cohesiveness.values())

    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise cluster
            continue
        normalized_size = cluster_sizes[cluster] / max_size
        normalized_cohesiveness = cluster_cohesiveness[cluster] / max_cohesiveness
        normalized_variance = cluster_variances[cluster]

        ranked_clusters[cluster] = []
        ranked_clusters[cluster].append(normalized_size)
        ranked_clusters[cluster].append(normalized_cohesiveness)
        ranked_clusters[cluster].append(normalized_variance)

    return ranked_clusters


def to_native_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {
            to_native_types(key): to_native_types(value) for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [to_native_types(element) for element in obj]
    else:
        return obj
