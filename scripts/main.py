"""
Airport Data Parsing.

* NOTES HERE *

"""
import networkx as nx
import os
import tempfile
import glob
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def parse_data(data_name='out.opsahl-openflights', option='r'):
    """Parse through data to create graph file."""
    import networkx as nx
    import os

    g = nx.DiGraph()
    edge_list = []

    with open(os.path.join('../data', data_name), 'rb') as file:
        data = file.read().decode().split('\n')
        for line in data[2:-1]:
            # Skip header lines
            node_direction = line.strip().split(' ')
            node_direction = tuple([int(x) for x in node_direction])
            edge_list.append(node_direction)
    g.add_edges_from(edge_list)
    if option == 'r':
        print(g.edges)
    elif option == 'w':
        write_graph(g)
    return


def write_graph(graph, save_dir='../data/saved_data', tmp_prefix='.'):
    """Write graph names, allowing for temporary graph names for subgraphs (Up to 10)."""
    tmp_files = glob.glob(os.path.join(save_dir, tmp_prefix + '*.out'))
    if len(tmp_files) >= 10:
        tmp_del_ask = 'There are 10 or more temp files, do you want to delete them before continuing? (Y/N): '
        tmp_del_ans = yes_or_no(tmp_del_ask)
        print('\n')
        if tmp_del_ans:
            iter_del = tqdm(tmp_files)
            for del_file in iter_del:
                iter_del.set_description(del_file)
                os.remove(del_file)
            print('All tmp files deleted!\n')
        else:
            print("Warning: Too many temp files might be bad!")

    in_result = input('What is the graphs name?: ')
    if in_result == '':
        tmp_file_ask = 'No input name will make a tempfile name. Is that okay? (Y/N): '
        tmp_file_ans = yes_or_no(tmp_file_ask)
        if tmp_file_ans:
            filename = tempfile.mktemp().replace('/tmp/', tmp_prefix)
        else:
            write_graph(graph, save_dir, tmp_prefix)
            return
    else:
        filename = in_result
    filename = filename + '.out'
    nx.write_gpickle(graph, os.path.join(save_dir, filename))
    print('Saved')
    return


def yes_or_no(input_question):
    """Simple yes or no function."""
    ans = input(input_question)
    ans = ans.lower()
    if ans == 'y':
        return True
    elif ans == 'n':
        return False
    else:
        input_question = '(Y/N)'
        return yes_or_no(input_question)


def load_graph(graph_name='main_graph', save_dir='../data/saved_data'):
    """Load up live."""
    import networkx as nx
    import os

    output = nx.read_gpickle(os.path.join(save_dir, graph_name + '.out'))

    return output


if __name__ == '__main__':
    # If first execution of code run: 
    parse_data(option='w')
    G = load_graph()
    """
    subgraph_value = 100
    sG = G.subgraph(range(subgraph_value))
    """
    pred = 'Predecess'
    succ = 'Success'
    nodes = G.nodes()
    for x in range(15)[1:]:
        node_max_threshold = 10 * x
        node_min_threshold = 2 * x

        dict_sub_main = {}
        for n in nodes:
            tmp_dict = {str(n): {pred: None, succ: None}}
            add_dict = False
            n_pred = len(list(G.predecessors(n)))
            n_succ = len(list(G.successors(n)))
            if n_pred >= node_min_threshold and n_pred <= node_max_threshold:
                tmp_dict[str(n)][pred] = n_pred
                add_dict = True
            if n_succ >= node_min_threshold and n_succ <= node_max_threshold:
                tmp_dict[str(n)][succ] = n_succ
                add_dict = True
            if add_dict:
                dict_sub_main.update(tmp_dict)
        subgraph_nodes = [int(x) for x in dict_sub_main.keys()]
        sG = G.subgraph(subgraph_nodes)
        dict_analysis = {
            'Edge_Ratio': [],
            'Counter_Obj': []
        }
        count = 0
        for node in sG:
            true_diff = G.degree(node)
            sub_diff = sG.degree(node)
            if sub_diff != 0:
                count += 1
            diff_ratio = sub_diff / true_diff
            dict_analysis['Edge_Ratio'].append((node, diff_ratio))
        np_ratios = np.array(list(zip(*dict_analysis['Edge_Ratio']))[1])
        # print(count / len(dict_analysis['Edge_Ratio']))
        label_string = str(node_min_threshold) + ' to ' + str(node_max_threshold) + ' (' + str(len(dict_analysis['Edge_Ratio'])) + ')'
        plt.hist(np_ratios, bins=45, alpha=0.5, edgecolor='k', label=label_string, density=True)
    plt.legend(loc='upper right')
    """
    hhist, hedges = np.histogram(x, bins=20)
    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)

    ph = figure(toolbar_location=None, plot_width=600, plot_height=200, x_range)



    curdoc().add_root(layout)
    curdoc().title = "Legend Select Histogram"
    """

    plt.show()