"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
fh = open("C:\\Users\\rapha\\OneDrive\\Bureau\\M2 MVA\\Altegrad\\Lab 2\\code\\datasets\\CA-HepTh.txt", "rb")
G = nx.read_edgelist(fh, comments= "#", delimiter= "\t")
fh.close()
print("nodes: ", G.number_of_nodes())
print("edges:", G.number_of_edges())
##################


############## Task 2

##################
print("The graph has", nx.number_connected_components(G), "connected components")

largest_cc = max(nx.connected_components(G), key = len)

print("The largest connected component in the graph has", len(largest_cc), "nodes")

#largest_cc = un array de noeuds
#subG = un objet de type graphe (comme G), qui est le sous graphe de G avec les edges associés aux noeufs largest_cc
subG = G.subgraph(largest_cc)
print("the largest connected component in the graph has", subG.number_of_edges(), "edges")
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
print("Minimum degree: ", np.min(degree_sequence))
print("Maximum degree: ", np.max(degree_sequence))
print("Mean degree: ", np.mean(degree_sequence))
print("Median degree: ", np.median(degree_sequence))
##################



############## Task 4

##################

hist = nx.degree_histogram(G)
degrees = range(len(hist))
fig, axs = plt.subplots(1, 2, figsize=(10, 20))

axs[0].plot(degrees, hist)
axs[0].set_xlabel('degree')
axs[0].set_ylabel('frequency')

axs[1].loglog(degrees, hist)
axs[1].set_xlabel('log(degree)')
axs[1].set_ylabel('log(frequency)')

plt.show()
##################




############## Task 5

##################
print("The global clustering coefficient of the graph is", nx.transitivity(G))
##################
