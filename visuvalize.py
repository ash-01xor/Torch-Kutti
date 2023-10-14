import graphviz

from torch_kutti import *

def visit_nodes(G, node):
    uid = str(id(node))
    G.node(uid, f"Tensor: ({str(node.data)}, grad: {str(node.grad.data) if node.grad is not None else 'None'}) ")
    if node._ctx:
        ctx_uid = str(id(node._ctx))
        G.node(ctx_uid, f"Context: {str(node._ctx.op.__name__)}")
        G.edge(uid, ctx_uid)
        for child in node._ctx.args:
            G.edge(ctx_uid, str(id(child)))
            visit_nodes(G, child)

def f(x):
    return x * x + x

if __name__ == "__main__":
    G = graphviz.Digraph(format='png')
    G.clear()

    x = Tensor([3.2])
    z = f(x)
    z.backward()

    visit_nodes(G, z)
    G.render(directory="vis", view=True)
    print(f"Z: {z}, grad: {z.grad}")