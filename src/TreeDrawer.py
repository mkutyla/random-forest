import Node as ND
import DecisionTree as DT
from treelib import Tree

tree = Tree()


def draw_tree(dt: DT.DecisionTree):
    """
    Draws a DT decision tree in readable format.

    Args:
        dt (DT.DecisionTree): a decision tree to draw
    """
    node = dt.root
    label = node.children_branches[0].label

    tree.create_node(f"{node.label} {label}", f"{node}")
    for branch in node.children_branches:
        __print_nodes(branch.child)
    tree.show()
    with open("tree.txt", "w"):
        tree.save2file("tree.txt")


def __print_nodes(node: ND.Node):
    if node.is_leaf():
        if "<=" in node.parent_branch.label:
            tree.create_node(
                f"(1) {node.cls}", f"{node}", parent=f"{node.parent_branch.parent}"
            )
        elif ">" in node.parent_branch.label:
            tree.create_node(
                f"(0) {node.cls}", f"{node}", parent=f"{node.parent_branch.parent}"
            )
        else:
            label = float(node.parent_branch.label[1:])  # cutting off "="
            tree.create_node(
                f"({label}) {node.cls}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )

    else:
        label = node.children_branches[0].label
        if "<=" in node.parent_branch.label:
            tree.create_node(
                f"(1) {node.label} {label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )
        elif ">" in node.parent_branch.label:
            tree.create_node(
                f"(0) {node.label} {label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )
        else:
            label = float(node.parent_branch.label[1:])  # cutting off "="
            tree.create_node(
                f"({label}) {node.label} {label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )

    for branch in node.children_branches:
        __print_nodes(branch.child)
