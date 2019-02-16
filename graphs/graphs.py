import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class ParentGraph:

    def __init__(self):
        self.state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def update_state(self, value):
        if self._state is None:
            self._state = value
        else:
            self._state = np.append(self._state, value, axis=0)

    def neighbors_of(self, node):
        return list(self[node].keys())

    def draw(self):
        pos = nx.spring_layout(self)
        plt.figure(figsize=(20, 10))

        # if edges are weighted, draw them
        try:
            weights = [self[u][v]['weight'] for u, v in self.edges]
            nx.draw_networkx_edges(self, pos, edges=self.edges(), width=weights)
        except KeyError:
            nx.draw_networkx_edges(self, pos, edges=self.edges())
        nx.draw_networkx_nodes(self, pos, node_size=20)
        plt.show()


class SmallWorldGraph(nx.Graph, ParentGraph):
    """
    Watts, Strogatz (1998), Humphries, Gurney (2008)
    """

    def __init__(self, n, k, p, connected=True):
        """
        :param n: (int)  number of nodes
        :param k: (int) number of edges
        :param p: (float) rewiring probability
        :param connected: (bool) should the graph be connected?
        """
        if connected:
            super(SmallWorldGraph, self).__init__(nx.connected_watts_strogatz_graph(n, k, p))
        else:
            super(SmallWorldGraph, self).__init__(nx.watts_strogatz_graph(n, k, p))


class ErdosReyniGraph(nx.Graph, ParentGraph):

    def __init__(self, n, p, seed=None, directed=False):
        """
        :param n: (int) number of nodes
        :param p: (float) probability of edge creation
        :param seed: RNG state
        :param directed: (bool) if True, return directed graph
        :return:
        """

        super(ErdosReyniGraph, self).__init__(nx.fast_gnp_random_graph(n, p, seed, directed))


class ScaleFreeGraph(nx.Graph, ParentGraph):

    def __init__(self, n, m=1):
        """
        :param n: (int) number of nodes
        :param m: (int) number of edges that a new node attaches to existing nodes
        """

        super(ScaleFreeGraph, self).__init__(nx.barabasi_albert_graph(n, m))


class GraphClone(nx.Graph, ParentGraph):

    def __init__(self, graph):
        """
        :param graph: nx.Graph constructor to clone from
        """
        super(GraphClone, self).__init__(incoming_graph_data=graph)


class DirectedGraph(nx.DiGraph, ParentGraph):

    def __init__(self):
        super(DirectedGraph, self).__init__()

