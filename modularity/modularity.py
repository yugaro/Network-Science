# import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import json
import community
from community import community_louvain

G = nx.Graph()

f = open('./data/my_sample_data.json')
data = json.load(f)
f.close()

data = {int(key): {int(sub_key): data[key][sub_key]
                   for sub_key in data[key].keys()} for key in data.keys()}
edge_list = [(key, sub_key, data[key][sub_key]) for key in data.keys()
             for sub_key in data[key].keys() if sub_key > key]

# create directed graph
G.add_weighted_edges_from(edge_list)

eigv_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

# implement of Clustering by modularity index
partition1 = community_louvain.best_partition(G, weight='weight')
partition2 = community_louvain.best_partition(
    G, partition={i: 0 if i < 10 else 1 for i in G.nodes()}, weight='weight')

print('Modularity Index of partition1:')
print(community.modularity(partition1, G))
print('Modularity Index of partition2:')
print(community.modularity(partition2, G))
