import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.tree.decision_tree import DecisionTree
from src.tree.node import Node


def format_float(i: float, n=2):
    return '{:g}'.format(float('{:.{p}g}'.format(i, p=n)))


def draw_tree(dt: DecisionTree) -> nx.Graph:
    g = nx.Graph()

    node = dt.root
    label = node.children_branches[0].label
    if "<" in label:
        value = float(label[2:])  # '<=' or '>.'
        value = format_float(value)
        g.add_node(node, text=f"{node.label}<={value}")
    elif ">" in label:
        value = float(label[1:])  # '<=' or '>.'
        value = format_float(value)
        g.add_node(node, text=f"{node.label}>{value}")
    else:
        g.add_node(node, text=f"{node.label}")

    for branch in node.children_branches:
        __add_nodes(g, node, branch.child)

    labels = nx.get_node_attributes(g, 'text')
    # pos = nx.circular_layout(g)
    # nx.draw(g, pos=pos, labels=labels, node_shape="s")

    center_node = node  # Or any other node to be in the center
    edge_nodes = set(g) - {node}
    pos = nx.circular_layout(g.subgraph(edge_nodes))
    pos[center_node] = np.array([0, 0])  # manually specify node position
    nx.draw(g, pos, labels=labels, node_shape="s")

    edge_labels = dict([((n1, n2), d['label'])
                        for n1, n2, d in g.edges(data=True)])

    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

    plt.show()


def __add_nodes(g: nx.Graph, parent: Node, node: Node):
    if node.is_leaf():
        if "<=" in node.parent_branch.label:
            g.add_node(node, text=f"{node.cls}")
            g.add_edge(parent, node, label='1')
        elif ">" in node.parent_branch.label:
            g.add_node(node, text=f"{node.cls}")
            g.add_edge(parent, node, label='0')
        else:
            label = node.parent_branch.label[1:]  # cutting off "="
            g.add_node(node, text=f"{node.cls}")
            g.add_edge(parent, node, label=f'{label}')
    else:
        label = node.children_branches[0].label
        if "<=" in node.parent_branch.label:
            value = float(node.parent_branch.label[2:])  # <=
            value = format_float(value)
            g.add_node(node, text=f"{node.label}<={value}")
            g.add_edge(parent, node, label='1')
        elif ">" in node.parent_branch.label:
            value = float(node.parent_branch.label[1:])  # >
            value = format_float(value)
            g.add_node(node, text=f"{node.label}>{value}")
            g.add_edge(parent, node, label='0')
        else:
            label = node.parent_branch.label[1:]
            g.add_node(node, text=f"{node.label}")
            g.add_edge(parent, node, label=f'{label}')

    for branch in node.children_branches:
        __add_nodes(g, node, branch.child)
