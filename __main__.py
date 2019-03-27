"""TBD."""
from .simulations.simulations import *
from pprint import pprint
import pickle
import os
from pathlib import Path

def gen_strong_graph(seed, init_size=1000, lower=20, upper=50):
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
        
        spec_const = np.ones(len(iterations.keys())) * dict_in[seed]['Spectral Const.']
        print(spec_const)
        # plt.semilogy(spec_const)
        plt.plot(plot_array)
        plt.figure()
        plt.plot(dict_in[seed]['Debug'], 'k--')
        plt.show()


if __name__ == '__main__':
    main_dict = {}

    seeds = [100]

    converg_t = 1000
    filename = os.path.join('cnets', 'data', 'saved_data', 'main_dict.pickle')
    overwrite = True
    if not Path(filename).exists() or overwrite:
        for seed in seeds:
            sim_obj = WaveEquationSimulation(store_dict=True)
            input = nx.random_lobster(500, 0.3, 0.5, seed=seed)
            # input = gen_strong_graph(seed)
            graph_in = clone(input)
            if graph_in is not None:
                main_dict[str(seed)] = sim_obj.run(graph_in,
                                                   n=converg_t,
                                                   checkpoint_amount=20,
                                                   c=0.5)
        with open(filename, 'wb') as handle:
            pickle.dump(main_dict, handle, pickle.HIGHEST_PROTOCOL)
        print('Saved Model')
    else:
        with open(filename, 'rb') as handle:
            main_dict = pickle.load(handle)

    

    # pprint(main_dict)
    plot_from_dict(main_dict)
    sim_obj.highlight_clusters()