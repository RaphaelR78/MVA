"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
fh = open("2024-2023-ENS\Altegrad\TP2\code\datasets\CA-HepTh.txt", "rb")
G = nx.read_edgelist(fh, comments= "#", delimiter= "\t")
fh.close()
print("nodes: ", G.number_of_nodes())
print("edges:", G.number_of_edges())
# your code here #
##################



############## Task 2

##################
print("The graph has", nx.number_connected_components(G), "connected components")

largest_cc = max(nx.connected_components(G), key = len)

print("The largest connected component in the graph has", len(largest_cc), "nodes")

subG = G.subgraph(largest_cc)
print("the largest connected component in the graph has", subG.number_of_edges(), "edges")
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
print("The minimal degree in the graph is", min(degree_sequence))
print("The maximal degree in the graph is", max(degree_sequence))
print("The median degree in the graph is", np.median(degree_sequence))
#print(len([x for x in degree_sequence if x<= 3]))
print("The mean degree in the graph is", np.mean(degree_sequence))
##################
# your code here #
##################



############## Task 4

##################
hist = nx.degree_histogram(G)

plt.figure("plot",figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(hist)
plt.xlabel('degree')
plt.ylabel('frequency')

plt.subplot(1,2,2)
plt.loglog(hist)
plt.xlabel('log(degree)')
plt.ylabel('log(frequency)')
#plt.show()
##################




############## Task 5

##################
print("The global clustering coefficient of the graph is", nx.transitivity(G))
##################
