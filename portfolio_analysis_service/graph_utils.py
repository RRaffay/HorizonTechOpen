import networkx as nx
import matplotlib.pyplot as plt


def construct_graph(related_entities_dict):
    G = nx.Graph()

    for stock, related_entities in related_entities_dict.items():
        G.add_node(stock, type="stock")

        for related_entity, description in related_entities.items():
            G.add_node(related_entity, type="entity")
            G.add_edge(stock, related_entity, description=description)

    return G


# Example Usage
# related_entities_dict = {
#     'AAPL': {'Supplier1': 'Provides chips', 'Supplier2': 'Provides screens'},
#     'GOOGL': {'Supplier1': 'Provides storage', 'Partner1': 'Collaboration in AI'},
# }

# G = construct_graph(related_entities_dict)

# # To visualize the graph
# nx.draw(G, with_labels=True)
# plt.show()
