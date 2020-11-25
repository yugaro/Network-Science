# import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import json
import community


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

# implement of Clustering
partition = community.best_partition(G, weight='weight')

# List the vertices separately for each community
part_com = [[] for _ in set(list(partition.values()))]
for key in partition.keys():
    part_com[partition[key]].append(key)

# List the vertices separately for each community by betweeness centrarlity
for part in part_com:
    G_part = nx.Graph()
    for edge in edge_list:
        if edge[0] in part and edge[1] in part:
            G_part.add_weighted_edges_from([edge])
    print(max(G_part.nodes(), key=lambda val: nx.betweenness_centrality(
        G_part, weight='weight')[val]))
