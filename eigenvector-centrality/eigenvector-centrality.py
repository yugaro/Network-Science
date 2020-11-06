import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

G.add_weighted_edges_from(edge_list)

plt.figure(figsize=(20, 20))

eigv_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

partition = community.best_partition(G, weight='weight')

node_size = np.array(list(eigv_cent.values())) * 10000
node_color = [partition[i] for i in G.nodes()]

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=400,
                       node_color=node_color, cmap=plt.cm.RdYlBu)
nx.draw_networkx_labels(G, pos, fontsize=15)
nx.draw_networkx_edges(G, pos, width=0.2)

plt.show()
