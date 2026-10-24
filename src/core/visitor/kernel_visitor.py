class KernelVisitor:
    """Walks Kernel IR by dispatching on the node's class name: Tile -> visit_Tile(node).

    Emmy's nodes have no accept(), so this lookup replaces double dispatch (like ast.NodeVisitor).
    """

    def __init__(self, info):
        self.info = info

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(f"no visitor for {type(node).__name__}")
        return method(node)

    def visit_Body(self, body):
        for stmt in body:
            self.visit(stmt)

    def visit_Tile(self, tile):
        ...
