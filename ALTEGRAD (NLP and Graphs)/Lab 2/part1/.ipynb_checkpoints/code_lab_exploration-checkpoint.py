"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
fh = open("\code\datasets\CA-HepTh.txt", "rb")
G = nx.read_edgelist(fh, comments= "#", delimiter= "\t")
fh.close()
print("nodes: ", G.number_of_nodes())
print("edges:", G.number_of_edges())
##################



############## Task 2

##################
# your code here #
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################



############## Task 4

##################
# your code here #
##################




############## Task 5

##################
# your code here #
##################
