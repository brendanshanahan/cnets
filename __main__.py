"""TBD."""
from .simulations.simulations import *
from pprint import pprint
import pickle
import os
from pathlib import Path

def gen_strong_graph(seed, init_size=200, p=0.1, lower=15, upper=100):
    """TBA."""
    graph = nx.erdos_renyi_graph(init_size, p, seed=seed, directed=True)
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
        # nx.draw(output)
        # plt.show()
        return output

def plot_from_dict(dict_in):
    conductance_list = []
    sp_list = []
    for seed in dict_in.keys():
        conductance_list.append(dict_in[seed]['Simulation Information']['Final Outcome']['Conductance'])
        sp_list.append(dict_in[seed]['Sparsity'])
        iterations = dict_in[seed]['Simulation Information']['Convergence']
        plot_array = np.zeros(len(iterations.keys()))
        for key in iterations.keys():
            plot_array[int(key.strip('Iteration-')) - 1] = iterations[key]['Conductance']
        
        spec_const = np.ones(len(iterations.keys())) * dict_in[seed]['Simulation Information']['Spectral Const.']
        # print(spec_const)
        # plt.semilogy(spec_const)
        # plt.plot(plot_array)
        # plt.figure()
        # plt.plot(dict_in[seed]['Debug'], 'k--')
    print(conductance_list)
    plt.figure()
    plt.plot(sp_list, conductance_list, 'ko')
    plt.show()


if __name__ == '__main__':
    main_dict = {}

    seeds = list(np.random.randint(100, size=25))
    p_list = list(np.random.uniform(0, 0.3, size=20))
    check_amount = 0
    converg_t = 7000
    filename = os.path.join('cnets', 'data', 'saved_data', 'main_dict.pickle')
    overwrite = False
    if not Path(filename).exists() or overwrite:
        for p in p_list:
            for seed in seeds:
                seed = int(seed)
                sim_obj = WaveEquationSimulation(store_dict=True)
                # input = nx.random_lobster(500, 0.3, 0.5, seed=seed)
                input = gen_strong_graph(seed, p=p)
                sparse_metric =  (input.number_of_edges() / (input.number_of_nodes()**2))
                graph_in = clone(input)
                if graph_in is not None:
                    main_dict[str(seed) + '_' + str(p)] = {'Simulation Information':
                                                sim_obj.run(graph_in,
                                                       n=converg_t,
                                                       checkpoint_amount=check_amount,
                                                       c=0.3),
                                            'Sparsity': sparse_metric,
                                            'Probability': p,
                                            'N': input.number_of_nodes()}
        with open(filename, 'wb') as handle:
            pickle.dump(main_dict, handle, pickle.HIGHEST_PROTOCOL)
        print('Saved Model')
    else:
        with open(filename, 'rb') as handle:
            main_dict = pickle.load(handle)
    # pprint(main_dict)
    plot_from_dict(main_dict)