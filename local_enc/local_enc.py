import os
from GraphRicciCurvature import OllivierRicci as oc
import pickle
import networkx as nx
import numpy as np
import stat
import torch

def computing_curvature(graph):
        a=oc._compute_ricci_curvature_edges(graph,weight="weight",edge_list=graph.edges())
        return a

def addin_attr(graph,curv_dict):
        nx.set_edge_attributes(graph,curv_dict,"RicciCurvature")
        return graph

def feature_vec(graph):
    nodes=list(graph.nodes())
    node_idx={
        n: i for i,n in enumerate(nodes)
      }

    feats=np.zeros((len(nodes),5))

    for n in nodes:
            
        i=node_idx[n]
        incident=[graph[i][nbr]["RicciCurvature"] for nbr in graph.neighbors(n) ]

        mean_c=incident.mean()
        min_c=incident.min()
        max_c=incident.max()
        std_c=incident.std()
        frac_neg=(incident<0).mean()
        feats[i] = [mean_c, min_c, max_c, std_c, frac_neg]
    
    return feats
    



class CurvatureMLP(torch.nn.Module):
    
    def __init__(self, in_dim=5, hidden_dim=32, out_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


                 
    
if __name__=='__main__':
    with open("graph.gpickle", "rb") as f:
        graph = pickle.load(f)

    curv_dict=computing_curvature(graph)
    graph=addin_attr(graph,curv_dict)
    feat=feature_vec(graph)
    mlp=CurvatureMLP()
    node_embed=mlp(feat)
    

