import networkx as nx
import numpy as np
import os
import math
import pickle
import argparse
from pathlib import Path
from preprocess.correlation import roi_correlation

def build_graph(file_path):
    fmri = np.load(file_path, allow_pickle=False)
    corr, _ = roi_correlation(fmri)  # saved arrays are [time, ROI]

    G = nx.from_numpy_array(corr)
    
    for node in G.nodes:
        G.nodes[node]["feature"] = corr[node]

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
    parser = argparse.ArgumentParser(description="Build a graph from one ROI time-series array.")
    parser.add_argument("timeseries_path", type=Path, help="Path to a [time, ROI] .npy file")
    parser.add_argument("--output", type=Path, default=Path("graph.gpickle"))
    args = parser.parse_args()

    graph=build_graph(args.timeseries_path)
    graph=drop_fraction_of_neighbors(graph)
    graph=changes_weight(graph,metric=None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(graph, f)

    print(graph)
    
