import numpy as np
import networkx as nx
from tqdm import tqdm
from graphs.graphs import *
import matplotlib.pyplot as plt
# import pdb


def relax(graph, node, laplacian, t):
    neighbors = graph.neighbors_of(node)
    sum = np.zeros(np.size(graph.nodes[node]['state']))

    for i in neighbors:
        sum += laplacian[i]


def draw_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(20, 10))

    # if edges are weighted, draw them
    try:
        weights = [graph[u][v]['weight'] for u, v in graph.edges]
        nx.draw_networkx_edges(graph, pos, edges=graph.edges(), width=weights)
    except KeyError:
        nx.draw_networkx_edges(graph, pos, edges=graph.edges())
    nx.draw_networkx_nodes(graph, pos, node_size=20)
    plt.show()


def plot_fft(graph):

    if graph.state is None:
        print("Graph state not initialized")
        return

    fft = np.fft.fft(graph.state, axis=0)
    ctr = (int)(len(fft)/2)
    fft = abs(np.append(fft[ctr:, :], fft[:ctr, :], axis=0))
    plt.figure(figsize=(20, 10))
    plt.plot(fft)
    plt.show()


def wave_equation(graph, n, c=0.1):
    """

    :param graph:
    :param n:
    :return:
    """

    # graph state should be uninitialized; if not, then clear it
    graph.state = None

    # initialize t = -1, 0 states
    init = np.random.uniform(size=(1, len(graph.nodes)))
    graph.update_state(np.append(init, init, axis=0))

    laplacian = nx.laplacian_matrix(graph).toarray()
    relaxation = tqdm(range(n))
    relaxation.set_description('Relaxation: ')

    for t in relaxation:
        # sum over all neighbors
        neighbors = np.dot((2 * np.identity(len(laplacian)) - (c**2) * laplacian), graph.state[t + 1])

        # t-1, t-2
        relaxation = - graph.state[t]

        graph.update_state([relaxation + neighbors])


def fft_peaks(graph):

    if graph.state is None:
        print('Need graph relaxation information for fft')
        return

    fft_relax = np.fft.fft(graph.state, axis=0)[1:]  # ignore first index ~ 0 frequency value

    # Empty initial conditions
    pr_increasing_nodes = np.array(-1)
    pr_decreasing_nodes = np.array(-1)
    peaks_array = np.zeros_like(fft_relax)

    dict_peak = {
            'Positive_Peaks': {},
            'Negative_Peaks': {}
        }

    for maxcheck in range(fft_relax.shape[0])[1:]:

        # Positive Max Check
        # Array of t - 1 values
        prior_value = fft_relax[maxcheck - 1, :]
        current_value = fft_relax[maxcheck, :]

        # presently decreasing values
        decreasing_nodes = np.where(prior_value > current_value)[0]
        peak_max_nodes = np.intersect1d(pr_increasing_nodes, decreasing_nodes)

        # Negative Min Check
        # presently increasing values
        increasing_nodes = np.where(prior_value < current_value)[0]
        peak_min_nodes = np.intersect1d(pr_decreasing_nodes, increasing_nodes)

        for peaks in peak_max_nodes:
            # Saving peak values for later
            peak_value = fft_relax[maxcheck, peaks]
            peaks_array[maxcheck, peaks] = peak_value

            if peaks not in dict_peak['Positive_Peaks'].keys():
                if peak_value > 0:
                    coeff = 1
                    if peaks not in dict_peak['Negative_Peaks'].keys():
                        dict_peak['Positive_Peaks'][peaks] = coeff

        for peaks in peak_min_nodes:
            peak_value = fft_relax[maxcheck, peaks]
            peaks_array[maxcheck, peaks] = peak_value

            if peaks not in dict_peak['Negative_Peaks'].keys():
                if peak_value < 0:
                    coeff = 0
                    if peaks not in dict_peak['Positive_Peaks'].keys():
                        dict_peak['Negative_Peaks'][peaks] = coeff


        pr_increasing_nodes = increasing_nodes
        pr_decreasing_nodes = decreasing_nodes
    return dict_peak


def highlight_clusters(graph, dict_input):
    # Wil be changed to take dict from some state of graph for output

    plt.figure(figsize=(20, 10))

    cluster_A = list(dict_input['Positive_Peaks'].keys())
    cluster_B = list(dict_input['Negative_Peaks'].keys())
    pos = nx.spring_layout(graph)

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=cluster_A,
                           node_color='r',
                           node_size=30)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=cluster_B,
                           node_color='b',
                           node_size=30)
    nx.draw_networkx_edges(graph, pos)
    plt.show()


