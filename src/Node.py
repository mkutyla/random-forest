from Branch import *


class Node:

    def __init__(self, parent_branch=None) -> None:
        self.parent_branch = parent_branch
        self.children_branches = list()
        self.label = None
        self.cls = None

    def add_child_branch(self, child: 'Node', split_at):
        branch = Branch(self, child, split_at)
        child.parent_branch = branch
        self.children_branches.append(branch)

    def is_leaf(self) -> bool:
        return self.cls is not None

    def transform_to_leaf(self, cls):
        self.cls = cls
        self.children_branches = list()
        self.label = None
