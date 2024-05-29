from django.http import JsonResponse
from .graph_utils import construct_graph


def analyze_portfolio(request):
    # Return JsonResponse saying feature not implemented
    return JsonResponse({"error": "Feature not implemented"})

    # Assume you have the related_entities_dict from the extract_related_entities view
    related_entities_dict = request.GET.get("related_entities_dict", {})

    G = construct_graph(related_entities_dict)

    # Perform analyses on the graph G and get results (e.g., key_nodes, insights)
    # key_nodes = list(nx.degree_centrality(G).keys())  # Replace with your own analysis
    insights = {}  # Replace with your own insights based on the graph

    # return JsonResponse({"key_nodes": key_nodes, "insights": insights})
    return insights


# Show the important nodes and hide the non important nodes (some threshold on centrality on some other measure)
#
#
