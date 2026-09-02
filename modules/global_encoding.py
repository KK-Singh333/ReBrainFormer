import os
import pickle
import torch
import torch_geometric.utils as utils

import numpy as np



class GlobalEncoding():
    def __init__(self,dim,use_edge_att=False,normalization=None):

        self.pos_enc_dim=dim
        self.normalisation=normalization
        self.use_edge_attr=use_edge_att


    def eigen_vecs(self,graph,gamma):
        edge_att=graph.edge_attr if self.use_edge_attr else None
        edge_index,edge_att=utils.get_laplacian(graph.edge_index,edge_att,normalization=self.normalisation,num_nodes=graph.num_nodes)
        L=utils.to_scipy_sparse_matrix(edge_index,edge_att)
        EigVal,EigVec=np.linalg.eig(L.toarray())
       
        exp_eigenvalues = np.exp(-gamma * EigVal) # sorts the eigen value 

        p =  (exp_eigenvalues) /np.sum(exp_eigenvalues)


        log_p = (np.log(p))


        embed=-1*p*log_p
        ## purt
        return (torch.from_numpy(embed).float())

    def purturbing_node(self, graph, gamma):
        
        original_embed = self.eigen_vecs(graph, gamma)  # shape: [num_nodes]
        original_score = original_embed.sum()           # scalar total entropy

        num_nodes = graph.num_nodes
        node_scores = torch.zeros(num_nodes)
        per_node_embed_diff = torch.zeros(num_nodes, num_nodes)  # optional, elementwise diffs

        edge_index = graph.edge_index
        edge_attr = graph.edge_attr if self.use_edge_attr else None

        for node in range(num_nodes):
            
            mask = (edge_index[0] != node) & (edge_index[1] != node)
            pert_edge_index = edge_index[:, mask]
            pert_edge_attr = edge_attr[mask] if edge_attr is not None else None

            
            pert_graph = graph.clone()
            pert_graph.edge_index = pert_edge_index
            if self.use_edge_attr:
                pert_graph.edge_attr = pert_edge_attr

            perturbed_embed = self.eigen_vecs(pert_graph, gamma)  # shape: [num_nodes]

            per_node_embed_diff[node] = perturbed_embed - original_embed

        return node_scores, per_node_embed_diff


  
if __name__=='__main__':
    

    with open("graph.gpickle", "rb") as f:
        graph = pickle.load(f)


    data=utils.from_networkx(graph)
    encoder = GlobalEncoding(
        dim=data.num_nodes,
        use_edge_att=True,
        normalization=None
    )

    embedding = encoder.eigen_vecs(data, gamma=1.0)

    print(embedding.shape)
    print(embedding)



