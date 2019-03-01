"""TBD."""
from .simulations.simulations import *
from pprint import pprint


def gen_strong_graph(seed, init_size=10000, lower=100, upper=50):
    """TBA."""
    graph = nx.scale_free_graph(init_size, seed=seed)
    strong_components = nx.algorithms.components.strongly_connected_components(graph)

    # Select components based on thresholds of acceptance
    selected_graph = [c for c in strong_components if len(c) > lower][0]

    if len(selected_graph) == 0:
        # If no well sized graph is created, pass over seed
        return None
    else:
        # Otherwise, return neede graph
        un_org_graph = graph.subgraph(selected_graph)
        old_map = sorted(un_org_graph.nodes)
        new_map = [n for n in range(len(old_map))]
        mapping = {old_map[x]: new_map[x] for x in range(len(new_map))}
        output = nx.relabel_nodes(un_org_graph, mapping)
        return output

def plot_from_dict(dict_in):
    for seed in dict_in.keys():
        iterations = dict_in[seed]['Convergence']
        plot_array = np.zeros(len(iterations.keys()))
        for key in iterations.keys():
            plot_array[int(key.strip('Iteration-')) - 1] = iterations[key]['Conductance']
        plot_array
        plt.semilogy(plot_array)
        plt.show()




if __name__ == '__main__':
    main_dict = {}

    seeds = [100, 215, 15]

    for seed in seeds:
        sim_obj = WaveEquationSimulation(store_dict=True)
        input = nx.random_lobster(500, 0.3, 0.5, seed=seed)
        # input = gen_strong_graph(seed)
        graph_in = clone(input)
        if graph_in is not None:
            main_dict[str(seed)] = sim_obj.run(graph_in,
                                               n=5000,
                                               checkpoint_amount=25)

    pprint(main_dict)
    plot_from_dict(main_dict)
