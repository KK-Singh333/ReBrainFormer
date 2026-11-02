import networkx as nx
import torch
from statsmodels.tsa.seasonal import STL
import pandas as pd
import pickle
import numpy as np
import argparse
from pathlib import Path

def createLouvainCommunity(G, community_size):
    '''Creates communities based on Louvain Algo, return list of sets'''
    communities=nx.community.louvain_communities(G, seed=123, resolution=community_size)
    n=G.number_of_nodes()
    community_mat = np.zeros((n, n), dtype=int)
    for i in range(0,n):
        for j in range(0,i+1):
            flag=0
            for community in communities:
                if i in community and j in community:
                    flag=1
                    break
            community_mat[i][j]=flag
            community_mat[j][i]=flag
    return communities,community_mat


def normaliseTimeSeries(communities, fmriTimeseries):
    '''Create a tensor of normalised timeseries per community'''
    community_tensor=[]
    node_idx_list=[]
    
    for community in communities:
        community_list = list(community)

        communityTimeseries = fmriTimeseries[community_list] 

        fmriTimeseries_tensor = torch.tensor(communityTimeseries, dtype=torch.float32)
        
        fmriTimeseries_norm_tensor = (fmriTimeseries_tensor - fmriTimeseries_tensor.min()) / (fmriTimeseries_tensor.max() - fmriTimeseries_tensor.min())
        
        community_tensor.append(fmriTimeseries_norm_tensor)
        node_idx_list.append(community_list)

    normalised_communities = community_tensor 
    
    return normalised_communities, node_idx_list


def mesoEmbed(normTimeSeries, communities):
    node_map = {}
    for community_idx, community in enumerate(normTimeSeries):
        community_ave = torch.mean(community, dim=0)
        community_stl = STL(
            pd.Series(community_ave.numpy()),
            period=12
        ).fit()
        trend_t = torch.tensor(
            community_stl.trend.values,
            dtype=torch.float32
        )
        for idx, vec in enumerate(community):
            node_idx = communities[community_idx][idx]
            node_stl = STL(
                pd.Series(vec.numpy()),
                period=12
            ).fit()
            vec_trend_t = torch.tensor(
                node_stl.trend.values,
                dtype=torch.float32
            )
            corr = torch.corrcoef(
                torch.stack([trend_t, vec_trend_t])
            )[0, 1].item()
            alpha = (corr + 1) / 2
            final_repr = (
                alpha * community_ave +
                (1 - alpha) * vec
            )
            final_repr = (final_repr - final_repr.mean()) / final_repr.std()
            node_map[node_idx] = final_repr
    return node_map

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Inspect mesoscopic features for one graph.")
    parser.add_argument("graph_path", type=Path, help="Path to a .gpickle graph")
    parser.add_argument("timeseries_path", type=Path, help="Path to a [time, ROI] .npy file")
    args = parser.parse_args()

    with args.graph_path.open("rb") as f:
        graph = pickle.load(f)


    communities, _ = createLouvainCommunity(graph, 1)

    time_series = np.load(args.timeseries_path, allow_pickle=False).T

    normalised_communities,node_idx_list=normaliseTimeSeries(communities,time_series)

    node_map=mesoEmbed(normalised_communities,node_idx_list)

    print(node_map)

    
