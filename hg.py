graph.ndata['feature'] = node_feat 


subg = dgl.in_subgraph(graph, seeds)

input_nodes = g.srcdata[dgl.NID]
output_nodes = g.dstdata[dgl.NID]

g = blocks[0]
# sub_g = dgl.edge_type_subgraph( g, [('_E')] )
# sub_g = dgl.node_type_subgraph( g, [('_N')] )
hg = dgl.to_homogeneous(sub_g, ndata=['feature'])

hg.edges()[0]

torch.cat([a,b],1)

g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
g.nodes['drug'].data['hv'] = th.zeros(3, 1)
g.nodes['disease'].data['hv'] = th.ones(3, 1)
