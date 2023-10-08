import graphviz

from torch_kutti import *

G = graphviz.Digraph(format='png')
G.clear()
def visit_nodes(G, node):
    uid = str(id(node))
    G.node(uid, f"Tensor: {str(node.data)}")
    
    if node._ctx:
        ctx_uid = str(id(node._ctx))
        G.node(ctx_uid, f"Context: {node._ctx.op.__class__.__name__}")
        G.edge(uid, ctx_uid)
        
        for child in node._ctx.args:
            G.edge(ctx_uid, str(id(child)))
            visit_nodes(G, child)

if __name__ == "__main__":
    x = Tensor([6])
    y = Tensor([9])    
    z = x+y
    visit_nodes(G,z)
    G.render(directory="vis",view=True)
    print(z)
    
    print(len(G.body))