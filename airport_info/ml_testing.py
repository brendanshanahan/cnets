import main

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE

if __name__ == '__main__':
    G = main.load_graph()
    n_list = G.nodes

    # Create Feature Sets Here
    # node in_degree
    in_fet = [len(list(G.predecessors(n))) for n in n_list]
    # node out_degree
    out_fet = [len(list(G.successors(n))) for n in n_list]

    X_data = np.array([in_fet, out_fet, ]).transpose()
    X_scaled = preprocessing.scale(X_data)
    print(X_scaled)
    plt.plot(X_scaled, 'ko')
    plt.show()
