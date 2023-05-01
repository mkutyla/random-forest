import Node


class Branch:

    def __init__(self, parent: Node, child: Node, label: str) -> None:
        self.parent = parent
        self.child = child
        self.label = label