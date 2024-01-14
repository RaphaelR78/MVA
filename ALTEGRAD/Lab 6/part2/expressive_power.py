"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4 #dimension of representations
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
Gs = [nx.cycle_graph(i) for i in range(10,20)]
##################




############## Task 5

##################
A = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
X = np.ones((A.shape[0],1))
idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()

A = sparse_mx_to_torch_sparse_tensor(A).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)

##################




############## Task 8
        
##################
model = GNN(1,hidden_dim,output_dim,neighbor_aggr,readout,dropout).to(device)
print(model(X,A,idx))
#for mean in readout: equality between representations. If we use sum different, 10 different representations


##################




############## Task 9
        
##################
G1 = nx.Graph()
G1.add_edges_from(nx.cycle_graph(3).edges())
G1.add_edges_from([(node + 3, (node + 1) % 3 + 3) for node in range(3)])

G2 = nx.cycle_graph(6)


##################
'''
G1 = nx.Graph()
G1.add_edges_from([(1, 2), (1, 3), (1, 4)])
G1.add_edge(2, 5)
G1.add_edge(5, 6)
G1.add_edge(3, 7)
G1.add_edge(7, 8)

G2 = nx.Graph()
G2.add_edges_from([(1, 2), (1, 3), (1, 4)])
G2.add_edge(4, 5)
G2.add_edge(5, 6)
G2.add_edge(6, 7)
G2.add_edge(7, 8)
print("G1 adj = ", nx.adjacency_matrix(G1).toarray())
print("G2 adj = ", nx.adjacency_matrix(G2).toarray())
'''
############## Task 10
        
##################
Gs = [G1,G2]
A = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
X = np.ones((A.shape[0],1))
idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()


A = sparse_mx_to_torch_sparse_tensor(A).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)
##################




############## Task 11
        
##################
model2 = GNN(1,hidden_dim,output_dim,"sum","sum",dropout).to(device)
print(model(X,A,idx))
##################



