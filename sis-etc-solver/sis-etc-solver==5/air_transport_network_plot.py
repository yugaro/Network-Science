import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community
# from IPython.display import SVG, display
np.random.seed(0)

# number of nodes (max: 50)
n = 50

# matrix of infection rates (air route matrix in case of the passengers and flights)
df_B = pd.read_csv('./data/US_Airport_Ad_Matrix.csv',
                   index_col=0, nrows=n, usecols=[i for i in range(n + 1)])
B = df_B.values

# define weighted multi-directed graph
G = nx.Graph()

# create SIS network
edge_list = [(i, j, B[i][j])
             for i in range(n) for j in range(n) if B[i][j] != 0]
G.add_weighted_edges_from(edge_list)

# define node size
eigv_cent = nx.eigenvector_centrality_numpy(G)
node_size = np.array([(size ** 4)
                      for size in list(eigv_cent.values())]) * 20000000

# define node color
partition = community.best_partition(G, weight='weight', resolution=1)
node_color = [partition[i] for i in G.nodes()]

# define edge width
width = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
width_std = 14 * (((width - min(width)) / (max(width) - min(width)))) + 0.5

# define label name
node_labels = {i: key for i, key in zip(np.array(range(n)), list(df_B.index))}

# plot graph
plt.figure(figsize=(40, 40))
# pos = nx.layout.spring_layout(G, k=0.9)
pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")

# set drawing nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_size,
    node_color=node_color,
    cmap=plt.cm.RdYlBu,
)

# set drawing labels
nx.draw_networkx_labels(
    G,
    pos,
    labels=node_labels,
    font_size=55
)

# set drawing edges
nx.draw_networkx_edges(
    G,
    pos,
    width=width_std
)

plt.gca().set_axis_off()
plt.savefig('./images/air_transport_network.png')
