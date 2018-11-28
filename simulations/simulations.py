import numpy as np
import networkx as nx
from tqdm import tqdm
from graphs.graphs import *
import matplotlib.pyplot as plt

import pdb


def relax(graph, node, laplacian, t):
    neighbors = graph.neighbors_of(node)
    sum = np.zeros(np.size(graph.nodes[node]['state']))

    for i in neighbors:
        sum += laplacian[i]

def draw_graph(graph):
    pos = nx.spring_layout(graph)
    # weights = [graph[u][v]['weight'] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, edges=graph.edges())
    nx.draw_networkx_nodes(graph, pos, node_size=20)
    plt.show()

def wave_equation(graph, n, c=0.1):
    """

    :param graph:
    :param n:
    :return:
    """

    # graph state should not be initialized yet
    try:
        assert graph.state is None
    except AssertionError:
        print("Graph already initialized; exiting method")
        return

    # initialize t = -1, 0 states
    init = np.random.uniform(size=(1, len(graph.nodes)))
    graph.update_state(np.append(init, init, axis=0))

    laplacian = nx.laplacian_matrix(graph).toarray()

    relaxation = tqdm(range(n))
    relaxation.set_description('Relaxation: ')

    # pdb.set_trace()

    for t in relaxation:
        # sum over all neighbors
        neighbors = np.dot((2*np.identity(len(laplacian)) - (c**2)*laplacian), graph.state[t+1])

        # t-1, t-2
        relaxation = - graph.state[t]

        graph.update_state([relaxation + neighbors])
