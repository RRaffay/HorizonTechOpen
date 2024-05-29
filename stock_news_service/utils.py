import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(entry, column_type):
    """
    Preprocesses a string entry from a dataframe column to extract relevant names.

    Args:
        entry (str): The string entry to preprocess.
        column_type (str): The column name, either 'location' or other.

    Returns:
        str: The preprocessed string containing only the extracted names.
    """
    if not isinstance(entry, str):
        return ""

    mentions = entry.split(";")

    if column_type == "location":
        names = [mention.split("#")[2] for mention in mentions]
    else:
        names = [mention.split(",")[0].replace(" ", "_") for mention in mentions]

    return " ".join(set(names))


def vectorize_columns(df, columns):
    """
    Vectorizes specified columns in a dataframe using TfidfVectorizer.

    Args:
        df (pandas.DataFrame): The input dataframe.
        columns (list): List of column names to vectorize.

    Returns:
        scipy.sparse.csr_matrix: The concatenated sparse matrix of vectorized columns.
    """
    vectorizer = TfidfVectorizer()
    vectors = []

    for column in columns:
        processed = df[column].apply(preprocess, args=(column,))
        X = vectorizer.fit_transform(processed)
        vectors.append(X)

    return hstack(vectors)


def cluster_data(data, min_cluster_size=5):
    """
    Clusters a scipy sparse matrix using HDBSCAN.

    Args:
        data (scipy sparse matrix): The sparse matrix to cluster.

    Returns:
        numpy.ndarray: The cluster labels for each sample.
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, gen_min_span_tree=True
    )
    return clusterer.fit_predict(data)


def extract_top_entities(column_name, df, n=5):
    """
    Extracts the top n entities by frequency from a column for each cluster.

    Args:
        column_name (str): The dataframe column to analyze.
        df (pandas.DataFrame): The clustered dataframe.
        n (int): Number of top entities to return per cluster.

    Returns:
        dict: Mapping of cluster IDs to list of top (entity, count) tuples.
    """

    clusters = {}
    for i in df["clusterdb"].unique():
        cluster_docs = df[df["clusterdb"] == i][column_name]
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

        clusters[i] = sorted(entity_count.items(), key=lambda x: x[1], reverse=True)[:n]
    return clusters


def silhouette_scores(df, data):
    """
    Computes silhouette scores for samples and adds them to the dataframe.

    Args:
        df (pandas.DataFrame): Dataframe containing cluster labels.
        data (scipy.sparse.csr_matrix): Clustered sparse matrix.

    Returns:
        pandas.DataFrame: Input df with silhouette scores added.
    """
    unique_labels = df["clusterdb"].unique()
    if len(unique_labels) > 1 and -1 in unique_labels:
        # Compute silhouette when there are more than one clusters and noise
        sample_silhouette_values = silhouette_samples(data.toarray(), df["clusterdb"])
        df["silhouette_value"] = sample_silhouette_values
    elif len(unique_labels) == 2 and -1 not in unique_labels:
        # Compute silhouette when there are exactly two clusters, no noise
        sample_silhouette_values = silhouette_samples(data.toarray(), df["clusterdb"])
        df["silhouette_value"] = sample_silhouette_values
    else:
        # When not possible to compute silhouette score, set it as None or a default value
        df["silhouette_value"] = None
    return df


# This needs works
# Need to figure out how to get the cluster health score to be a better measure of what
# we want to see.
# Just using the silhouette score for now.


def rank_clusters(df):
    """
    Ranks clusters based on silhouette score, size, and entity frequency.

    Args:
        df (pandas.DataFrame): Clustered dataframe.

    Returns:
        dict: Mapping of cluster IDs to normalized ranking score.
    """
    # Calculate silhouette scores and cluster sizes
    silhouette_scores = df.groupby("clusterdb")["silhouette_value"].mean()
    cluster_sizes = df["clusterdb"].value_counts()

    # For entity frequency, we will take the sum of the top entities for each cluster
    entity_freq = {}
    columns = ["V2Persons", "V2Themes", "V2Organizations", "V2Locations"]
    for column in columns:
        entities_clusters = extract_top_entities(column, df)
        for cluster, entities in entities_clusters.items():
            entity_freq[cluster] = entity_freq.get(cluster, 0) + sum(
                [entity[1] for entity in entities]
            )

    # Normalize each ranking factor
    max_sil = max(silhouette_scores)
    max_size = max(cluster_sizes)
    max_entity = max(entity_freq.values())

    normalized_sil = {k: v / max_sil for k, v in silhouette_scores.items()}
    normalized_size = {k: v / max_size for k, v in cluster_sizes.items()}
    normalized_entity = {k: v / max_entity for k, v in entity_freq.items()}

    # Sum normalized scores
    cluster_ranks = {}
    for cluster in df["clusterdb"].unique():
        cluster_ranks[cluster] = (
            normalized_sil[cluster]
            # * normalized_size[cluster]
            # * normalized_entity[cluster]
        )

    return cluster_ranks


def extract_diverse_top_articles(
    cluster_vectors, centroid, n=5, similarity_threshold=0.95
):
    """
    Extracts top n articles that are not too similar to each other.

    Args:
        cluster_vectors (scipy.sparse.csr_matrix): Vectors of articles within a cluster.
        centroid (numpy.ndarray): Centroid of the cluster.
        n (int): Number of top articles to select.
        similarity_threshold (float): Threshold for similarity between selected articles.

    Returns:
        list: Indices of the top n diverse articles.
    """

    # Rank articles based on their distance to the centroid
    distances_to_centroid = cosine_similarity(cluster_vectors, centroid.reshape(1, -1))
    sorted_indices = distances_to_centroid.argsort(axis=0)[::-1].flatten()

    selected_indices = [sorted_indices[0]]  # start with the closest article
    for idx in sorted_indices[1:]:
        is_too_similar = False
        for selected_idx in selected_indices:
            if (
                cosine_similarity(cluster_vectors[idx], cluster_vectors[selected_idx])[
                    0
                ][0]
                > similarity_threshold
            ):
                is_too_similar = True
                break

        if not is_too_similar:
            selected_indices.append(idx)

        if len(selected_indices) == n:
            break

    return selected_indices


def extract_important_articles2(
    df, X_all, n=5, ranked_clusters=None, top_k=5, health_cutoff=0
):
    """
    Extracts the top articles and entities for each cluster.

    Args:
        df (pandas.DataFrame): Clustered dataframe.
        X_all (scipy.sparse.csr_matrix): Full clustered sparse matrix.
        n (int): Number of top articles per cluster.
        ranked_clusters (dict): Cluster rankings, if available.
        top_k (int): Number of top entities per cluster.

    Returns:
       dict: Mapping of cluster IDs to extracted info.
    """
    unique_clusters = df["clusterdb"].unique()
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

        cluster_vectors = X_all[df["clusterdb"] == cluster]

        # If there are no vectors in the cluster, continue to the next iteration
        if cluster_vectors.shape[0] == 0:
            continue

        # Calculate cluster's centroid
        centroid = cluster_vectors.mean(axis=0).A[
            0
        ]  # Convert to dense array after mean calculation

        # Calculate similarities within the cluster
        intra_cluster_similarities = cosine_similarity(cluster_vectors)

        # Check if all articles in the cluster have similarity > 0.9 with each other
        if (intra_cluster_similarities > 0.9).all():
            # If yes, simply select the first article as the representative
            top_article_indices = [0]
        else:
            # Otherwise, rank articles based on their distance to the centroid
            top_article_indices = extract_diverse_top_articles(
                cluster_vectors, centroid, n=n
            )

        # Extract meta-data for the top articles
        top_articles_metadata = df[df["clusterdb"] == cluster].iloc[top_article_indices]

        # Extract the most frequently occurring entities
        entities_clusters = extract_top_entities(
            "V2Themes", df[df["clusterdb"] == cluster], top_k
        )
        top_themes = entities_clusters[cluster]

        entities_clusters = extract_top_entities(
            "V2Persons", df[df["clusterdb"] == cluster], top_k
        )
        top_persons = entities_clusters[cluster]

        entities_clusters = extract_top_entities(
            "V2Organizations", df[df["clusterdb"] == cluster], top_k
        )
        top_orgs = entities_clusters[cluster]

        entities_clusters = extract_top_entities(
            "V2Locations", df[df["clusterdb"] == cluster], top_k
        )
        top_locs = entities_clusters[cluster]

        cluster_info[cluster] = {
            "top_articles": top_articles_metadata["DocumentIdentifier"].tolist(),
            "top_themes": top_themes,
            "top_persons": top_persons,
            "top_orgs": top_orgs,
            "top_locs": top_locs,
            "cluster_health": cluster_health,
        }

    return cluster_info


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
