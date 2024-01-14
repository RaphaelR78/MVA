"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(1, walk_length):
        neighbors = list(G.neighbors(walk[i-1]))
        random_neighbor = neighbors[randint(0, len(neighbors) - 1)]
        walk.append(random_neighbor)

        
    walk = [str(node) for node in walk]
    return walk


G = nx.Graph()
G.add_node(1);G.add_node(2);G.add_node(3)
G.add_edge(1, 2);G.add_edge(1, 3)
#nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
#plt.show()




############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    for v in G.nodes():
        for i in range(num_walks):
                walks.append(random_walk(G,v,walk_length))

    permuted_walks = np.random.permutation(walks)
    return permuted_walks.tolist()

#print(generate_walks(G,5,4))


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
