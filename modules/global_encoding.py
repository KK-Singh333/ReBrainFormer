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
        idx=EigVal.argsort()
        exp_eigenvalues = np.exp(-gamma * EigVal[idx]) # sorts the eigen value 

        p =  np.diag(exp_eigenvalues) /np.sum(exp_eigenvalues)


        log_p = np.diag(np.log(p))


        embed=p*log_p
        return (torch.from_numpy(embed[:,1:self.pos_enc_dim+1]).float())
    

        
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



