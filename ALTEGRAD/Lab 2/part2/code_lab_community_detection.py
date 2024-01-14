"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    A = nx.adjacency_matrix(G)
    D_inv = np.diag([1/G.degree(node) for node in G.nodes()])
    I = np.identity(D_inv.shape[0])
    L = I - D_inv@A
    _, U = eigs(L, k=k, which='SR')
    U = np.real(U)

    kmeans = KMeans(n_clusters=k,random_state=0).fit(U)

    clustering ={}

    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    
    return clustering





############## Task 7
fh = open("C:\\Users\\rapha\\OneDrive\\Bureau\\M2 MVA\\Altegrad\\Lab 2\\code\\datasets\\CA-HepTh.txt", "rb")
G = nx.read_edgelist(fh, comments= "#", delimiter= "\t")
fh.close()
subG = G.subgraph(max(nx.connected_components(G), key = len))
clustering_of_largest_cc = spectral_clustering(subG, 50)
#print(clustering_of_largest_cc)



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    modularity = 0
    m = G.number_of_edges()
    nc = len(set(clustering.values()))

    for i in range(nc):
        nodes_i = [nodes for nodes,clust  in clustering.items() if clust == i]
        community = G.subgraph(nodes_i)
        lc = community.number_of_edges()
        dc = np.sum([G.degree(node) for node in nodes_i])
        modularity += lc/m - (dc/(2*m))**2
    
    return modularity


############## Task 9

m1 = modularity(G,clustering_of_largest_cc)
m2 = modularity(G,{node: randint(0, 49) for node in G.nodes()})
print("Modularity for spectral clustering: ", m1)
print("Modularity for random clustering: ", m2)









