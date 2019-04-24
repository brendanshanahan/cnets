"""TBD."""
from .simulations.simulations import *
from descartes import PolygonPatch
from pprint import pprint
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import triangulate
import pickle
import os
from pathlib import Path
import statistics
import random
import matplotlib
from matplotlib.collections import LineCollection

def gen_strong_graph(seed, init_size=200, p=0.1, lower=15, upper=100):
    """TBA."""
    graph = nx.erdos_renyi_graph(init_size, p, seed=seed, directed=True)
    strong_components = nx.algorithms.components.strongly_connected_components(graph)

    # Select components based on thresholds of acceptance
    selected_graph = [c for c in strong_components if len(c) > lower]
    if not selected_graph:
        return None
    else:
        selected_graph = selected_graph[0]

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

def plot_from_dict(dict_in, fig, c, ax):
    conductance_list = []
    sp_list = []
    p_list = []
    for seed in dict_in.keys():
        conduc = dict_in[seed]['Simulation Information']['Final Outcome']['Conductance']
        if conduc > 0 and conduc < 4:
            conductance_list.append(conduc)
            sp_list.append(dict_in[seed]['Sparsity'])
        p_list.append(dict_in[seed]['Probability'])
        iterations = dict_in[seed]['Simulation Information']['Convergence']
        plot_array = np.zeros(len(iterations.keys()))
        for key in iterations.keys():
            plot_array[int(key.strip('Iteration-')) - 1] = iterations[key]['Conductance']
        # plt.plot(plot_array)
        # plt.figure()
        # plt.plot(dict_in[seed]['Simulation Information']['Debug'], 'k--')
    cond_avg = sum(conductance_list) / len(conductance_list)
    point_list = list(zip(sp_list, conductance_list))
    poly = MultiPoint(point_list).convex_hull.buffer(0.01)
    r = lambda: random.randint(0,255)
    hex_color = ('#%02X%02X%02X' % (r(),r(),r()))
    patch = PolygonPatch(poly, fc=hex_color,
                         alpha=0.2,
                         ec='#000000', fill=True)
    # ax.add_patch(patch)
    ax.plot(sp_list, conductance_list, hex_color, marker='x', linestyle='None', markersize=5, label=str(c))
    ax.set_ylabel('Conductance', fontweight='bold',fontsize='large')
    ax.set_xlabel('Sparseness of Graph (Edges / Nodes)', fontweight='bold',fontsize='large')
    ax.set_title('Final Conductance for Different c coefficients', fontweight='bold',fontsize='large')
    return cond_avg, statistics.stdev(conductance_list)


if __name__ == '__main__':
    font = {'size': 18}

    matplotlib.rc('font', **font)
    main_dict = {}
    c_avg_list = []
    c_std_list = []
    c_list = [0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.06, 0.05, 0.02]
    fig, (ax, ax2) = plt.subplots(1, 2)
    np.random.seed(10)
    for c in c_list:
        seeds = list(np.random.randint(100, size=10))
        p_list = list(np.random.uniform(0, 0.2, size=50))
        check_amount = 0
        converg_t = 5000
        full_name = '_'.join(['main_dict', str(c).replace('.', '_'), str(converg_t)]) + '.pickle'
        filename = os.path.join('cnets', 'data', 'saved_data', full_name)
        overwrite = False
        if not Path(filename).exists() or overwrite:
            for p in p_list:
                for seed in seeds:
                    seed = int(seed)
                    sim_obj = WaveEquationSimulation(store_dict=True)
                    # input = nx.random_lobster(500, 0.3, 0.5, seed=seed)
                    input = gen_strong_graph(seed, p=p)
                    if input is None:
                        print('Skip this value')
                        continue
                    sparse_metric = (input.number_of_edges() / (input.number_of_nodes()**2))
                    graph_in = clone(input)
                    if graph_in is not None:
                        main_dict[str(seed) + '_' + str(p)] = {
                            'Simulation Information':
                                sim_obj.run(graph_in,
                                            n=converg_t,
                                            checkpoint_amount=check_amount,
                                            c=c),
                            'Sparsity': sparse_metric,
                            'Probability': p,
                            'N': input.number_of_nodes(),
                            'c': c,
                            'Iterations': converg_t}
            with open(filename, 'wb') as handle:
                pickle.dump(main_dict, handle, pickle.HIGHEST_PROTOCOL)
            print('Saved Model')
        else:
            with open(filename, 'rb') as handle:
                main_dict = pickle.load(handle)
        # pprint(main_dict)
        c_avg, c_std = plot_from_dict(main_dict, fig, c, ax)
        c_avg_list.append(c_avg)
        c_std_list.append(c_std)
    ax.legend(title='Values of c')
    ax2.errorbar(c_list, c_avg_list, yerr=c_std_list, marker='o', ecolor='r')
    ax2.set_xlabel('c value', fontweight='bold', fontsize='large')
    ax2.set_ylabel('Conductance', fontweight='bold',fontsize='large')
    ax2.set_title('Mean Conductance Values for Different c', fontweight='bold')
    plt.show()