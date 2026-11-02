import os
import pickle
import torch
import torch_geometric.utils as utils




class GlobalEncoding():
    def __init__(self,dim,use_edge_attr=False,normalization=None):

        self.pos_enc_dim=dim
        self.normalisation=normalization
        self.use_edge_attr=use_edge_attr


    def eigen_vecs(self,graph,gamma):
        if self.normalisation == "rw":
            raise ValueError("Random-walk Laplacian is not symmetric; use None or 'sym'")
        edge_att=graph.edge_attr if self.use_edge_attr else None
        edge_index,edge_att=utils.get_laplacian(graph.edge_index,edge_att,normalization=self.normalisation,num_nodes=graph.num_nodes)
        laplacian = edge_att.new_zeros((graph.num_nodes, graph.num_nodes))
        laplacian.index_put_((edge_index[0], edge_index[1]), edge_att, accumulate=True)
        dtype = torch.float32 if laplacian.device.type == "cuda" else torch.float64
        eigenvalues = torch.linalg.eigvalsh(laplacian.to(dtype))

        # Same formula as before: p_k = exp(-gamma * lambda_k) / sum_j exp(...),
        # and the returned vector contains -p_k * log(p_k).
        p = torch.softmax(-gamma * eigenvalues, dim=0)
        return (-torch.xlogy(p, p)).to(torch.float32)

    def purturbing_node(self, graph, gamma):
        
        original_embed = self.eigen_vecs(graph, gamma)  # shape: [num_nodes]
        original_score = original_embed.sum()           # scalar total entropy

        num_nodes = graph.num_nodes
        per_node_embed_diff = original_embed.new_zeros(num_nodes, num_nodes)

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

        return float(torch.trace(per_node_embed_diff).item()), per_node_embed_diff


  
if __name__=='__main__':
    

    with open("graph.gpickle", "rb") as f:
        graph = pickle.load(f)


    data=utils.from_networkx(graph)
    encoder = GlobalEncoding(
        dim=data.num_nodes,
        use_edge_attr=True,
        normalization=None
    )

    embedding = encoder.eigen_vecs(data, gamma=1.0)

    print(embedding.shape)
    print(embedding)



