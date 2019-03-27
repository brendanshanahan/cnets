import numpy as np
import networkx as nx
from tqdm import tqdm
from ..graphs.graphs import *
import matplotlib.pyplot as plt
import copy


class WaveEquationSimulation(object):

    def __init__(self, store_dict=False):
        """
        initialize a simulation with no input arguments. simulations supersede graphs in a sense, i.e.
        a simulation can act on multiple distinct graphs, but a graph has to be "reset" between simulations
        """

        self.graph = None
        self.peaks = None
        self.all_graphs = {}
        self.store_dict = store_dict

    def store_simulation_data(self, graph, **data):
        """
        once we run the simulation we (probably) don't need to keep a copy of the graph, so we can
        collect whatever relevant data we're interested in and store just that

        :param graph: graph instance which the simulation has already ran on
        :param data: it probably makes sense to store all the data we want in a set
        """

        self.all_graphs[hash(graph)] = data

    def run(self, graph, n=2000, c=0.1, checkpoint_amount=10):
        """
        :param graph: not a networkx graph, but an instance of the wrapper class in graphs.py
        :param n: number of wave equation time steps
        :param c: wave speed
        """
        if self.store_dict:
            # Main dictionary to be stored in class
            dict_store_graph_data = {
                'Convergence': None,
                'Spectral Const.': 0.001,
                'Complete Search.': 'N/A  (For now)',
                'Final Outcome': None,
                'Debug': []
            }
        # Set up foundation for comparisons later

        ## Spectral Clustering Values


        # graph state should be uninitialized; if not, then clear it
        graph.state = np.zeros((n+2, len(graph.nodes)))

        # initialize t = -1, 0 states
        init = np.random.uniform(size=(2, len(graph.nodes)))
        graph.state[:2, :] = init

        if graph.is_directed():
            laplacian = np.asarray(nx.directed_laplacian_matrix(graph))
        else:
            laplacian = nx.laplacian_matrix(graph).toarray()
        relaxation_it = tqdm(range(n))
        relaxation_it.set_description('Starting ...')

        # Dictionary for all checkpoint values
        checkpoint_dict = {}

        for t in relaxation_it:
            # sum over all neighbors
            neighbors = np.dot((2 * np.identity(len(laplacian)) - (c**2) * laplacian), graph.state[t + 1, :])

            # t-1, t-2
            relaxation = graph.state[t, :]

            graph.state[t + 2, :] = neighbors - relaxation

            # Checkpoint stopping at even intervals
            if (t+1) % int(n / checkpoint_amount) == 0:

                # Update init graph state, update console output
                self.graph = copy.deepcopy(graph)
                self.graph.state = copy.deepcopy(graph.state[:t, :])
                self.peaks =  self.__fft_peaks()
                self.draw_partitions()

                dict_store_graph_data['Debug'].append(len(self.__fft_peaks().keys()))
                # self.peaks = self.__fft_peaks()
                # self.draw_partitions()

                checkpoint_count = int(t / (n / checkpoint_amount)) + 1
                relaxation_it.set_description('Passing checkpoint ' + str(checkpoint_count) + ' / ' + str(checkpoint_amount))

                checkpoint_key = '-'.join(['Iteration', str(checkpoint_count)])
                checkpoint_dict[checkpoint_key] = self.__test_graph()

        # if a WaveEquationSimulation acts on a single graph at a time, then functions below like
        # highlight_clusters, etc only make sense inside WaveEquationSimulation if it maintains a local copy of
        # the current graph after running the simulation

        self.graph = graph
        self.peaks = self.__fft_peaks()
        if self.store_dict:
            dict_store_graph_data['Convergence'] = checkpoint_dict
            dict_store_graph_data['Final Outcome'] = self.__test_graph()
            return dict_store_graph_data
        else:
            self.peaks = self.__fft_peaks()
            return

    def __fft_peaks(self):
        if self.graph.state is None:
            print('Need graph relaxation information for fft')
            return

        fft_relax = np.fft.fft(self.graph.state, axis=0)[1:]  # ignore first index ~ 0 frequency value

        # Empty initial conditions
        pr_increasing_nodes = np.array(-1)
        pr_decreasing_nodes = np.array(-1)
        peaks_array = np.zeros_like(fft_relax)

        peaks = {}

        # can this be made any simpler/easier to understand?
        for maxcheck in range(1, fft_relax.shape[0]):

            # Positive Max Check
            # Array of t-1 values
            prior_value = fft_relax[maxcheck-1, :]
            current_value = fft_relax[maxcheck, :]

            # presently decreasing values
            decreasing_nodes = np.where(prior_value > current_value)[0]
            peak_max_nodes = np.intersect1d(pr_increasing_nodes, decreasing_nodes)

            # Negative Min Check
            # presently increasing values
            increasing_nodes = np.where(prior_value < current_value)[0]
            peak_min_nodes = np.intersect1d(pr_decreasing_nodes, increasing_nodes)

            for node in peak_max_nodes:
                # Saving peak values for later
                peak_value = fft_relax[maxcheck, node]
                peaks_array[maxcheck, node] = peak_value

                if peak_value > 0 and (node not in peaks or peaks[node] is not 0):
                        peaks[node] = 1

            for node in peak_min_nodes:
                peak_value = fft_relax[maxcheck, node]
                peaks_array[maxcheck, node] = peak_value

                if peak_value < 0 and (node not in peaks or peaks[node] is not 1):
                        peaks[node] = 0

            pr_increasing_nodes = increasing_nodes
            pr_decreasing_nodes = decreasing_nodes
        return peaks

    def __test_graph(self):
        assert self.graph.state is not None, "Graph state is not initialized"

        checkpoint_cluster = self.__fft_peaks()


        # Add more value later, just checking conductance now.
        dict_save = {
            'Conductance': self.__conductance(checkpoint_cluster)
        }
        # print(dict_save)
        return dict_save

    def __conductance(self, peaks, cluster_size=2):
        clusters = []

        for it in range(cluster_size):
            empty = [node for node in peaks.keys() if peaks[node] == it]
            if not empty:
                return -1
            clusters.append(empty)
        return nx.algorithms.cuts.conductance(self.graph, *clusters)

    def highlight_clusters(self):
        """TODO: take dict from some state of graph for output."""
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(20, 10))
        pos_cluster = list(int(k) for (k, v) in self.peaks.items() if v == 1)
        neg_cluster = list(int(k) for (k, v) in self.peaks.items() if v == 0)
        labels = {k: str(v) for (k, v) in self.peaks.items()}

        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=pos_cluster,
                               node_color='r',
                               node_size=50)
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=neg_cluster,
                               node_color='b',
                               node_size=50)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos, labels=labels)
        plt.show()

    def draw_partitions(self):
        """TODO: make this not stupid."""
        positive_nodes = list(k for (k, v) in self.peaks.items() if v == 1)
        negative_nodes = list(k for (k, v) in self.peaks.items() if v == 0)

        pos = nx.bipartite_layout(self.graph, positive_nodes)
        plt.figure(figsize=(20, 10))

        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=positive_nodes,
                               node_color='r',
                               node_size=50)
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=negative_nodes,
                               node_color='b',
                               node_size=50)
        nx.draw_networkx_edges(self.graph, pos, edges=self.graph.edges())
        plt.show()

    def plot_fft(self):
        if self.graph.state is None:
            print("Graph state not initialized")
            return

        fft = np.fft.fft(self.graph.state, axis=0)[1:]
        ctr = int(len(fft) / 2)
        fft = np.append(fft[ctr:, :], fft[:ctr, :], axis=0)
        plt.figure(figsize=(20, 10))
        plt.plot(fft)
        plt.show()
