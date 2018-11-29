import networkx as nx
import numpy as np


class SmallWorldGraph(nx.Graph):
    """
    small-world network as defined by Watts, Strogatz (1998), Humphries, Gurney (2008)
    """

    def __init__(self, n, k, p, connected=True):
        """
        :param n: (int)  number of nodes
        :param k: (int) number of edges
        :param p: (float) rewiring probability
        :param connected: (bool) should the graph be connected?
        """
        if connected:
            super().__init__(nx.connected_watts_strogatz_graph(n, k, p))
        else:
            super().__init__(nx.watts_strogatz_graph(n, k, p))

        self._state = None

        # for node in self.nodes():
        #     self.nodes[node]['state'] = []

    @property
    def state(self):
        return self._state
        # return self.nodes[node]['state']

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


class ErdosReyniGraph(nx.Graph):
    """

    """

    def __init__(self, n, p, seed=None, directed=False):
        """
        :param n:
        :param p:
        :param seed:
        :param directed:
        :return:
        """

        super().__init__(nx.fast_gnp_random_graph(n, p, seed, directed))

        for node in self.nodes():
            self.nodes[node]['state'] = []


class ScaleFreeGraph(nx.Graph):
    """

    """

    def __init__(self, n, m=1):
        """
        :param n: (int) number of nodes
        :param m: (int) number of edges that a new node attaches to existing nodes
        """

        super().__init__(nx.barabasi_albert_graph(n, m))

        for node in self.nodes():
            self.nodes[node]['state'] = []

class GraphClone(nx.Graph):

    def __init__(self, graph):
        super().__init__(incoming_graph_data=graph)

        self._state = None

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