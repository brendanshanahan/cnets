import requests

# parse incoming power grid data
def parse_grid(grid):
    response = requests.get(grid).text.split('\n')[:-1]
    edges = []

    for edge in response:
        edges.append([int(node) for node in edge.split(' ')])

    return edges
