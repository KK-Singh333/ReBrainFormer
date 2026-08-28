import networkx as nx
import numpy as np
import os
import math
import pickle

def build_graph(file_path):
    
    fmri = np.load(file_path)
    fmri = fmri.squeeze() 

    if fmri.shape[0] >= fmri.shape[1]:
        fmri = fmri.T

    corr = np.corrcoef(fmri) # making the correlation matrix
    np.fill_diagonal(corr, 0) # removing the self loops

    G = nx.from_numpy_array(corr)

    return G



def drop_fraction_of_neighbors(G, a=2.0, weight_attr='weight'):

    G_sparse = G.copy()
    edges_to_remove = set()
    for node in G.nodes():
        
        neighbors = G[node]
        k = len(neighbors)
        
        if k == 0:
            continue  

        num_to_drop = int(k / a)

        sorted_neighbors = sorted(
            neighbors.items(), 
            key=lambda item: abs(item[1].get(weight_attr, 0))
        )
        

        for i in range(num_to_drop):
            weak_neighbor = sorted_neighbors[i][0]

            edge_to_drop = tuple(sorted((node, weak_neighbor)))
            edges_to_remove.add(edge_to_drop)
            

    G_sparse.remove_edges_from(edges_to_remove)


    
    return G_sparse

def changes_weight(graph,metric=None):
    edges=graph.edges()

    
    for c in edges:
        if metric==None:
            graph[c[0]][c[1]]['weight']=(1/(graph[c[0]][c[1]]['weight']+1e-4))
        elif metric=='root_dis':
            graph[c[0]][c[1]]['weight']=math.sqrt(2*(1-graph[c[0]][c[1]]['weight']))
        elif metric=='dis':
            graph[c[0]][c[1]]['weight']=(1-graph[c[0]][c[1]]['weight'])

    return graph



if __name__=='__main__':
    graph=build_graph("/home/snottyunhunter/ReBrainFormer/test_data_npy/Pitt_0050003_func_preproc.nii.npy")
    graph=drop_fraction_of_neighbors(graph)
    graph=changes_weight(graph,metric=None)

    with open("graph.gpickle", "wb") as f:
        pickle.dump(graph, f)

    print(graph)
    
