import networkx as nx
import torch
from statsmodels.tsa.seasonal import STL
import pandas as pd

def createLouvainCommunity(G, community_size):
    '''Creates communities based on Louvain Algo, return list of sets'''
    communities=nx.community.louvain_communities(G, seed=123, resolution=community_size)
    return communities


def normaliseTimeSeries(communities, fmriTimeseries):
    '''Create a tensor of normalised timeseries per community'''
    community_tensor=[]
    node_idx_list=[]
    for community in communities:
        community_list=list(community)
        communityTimeseries=fmriTimeseries[community_list]
        fmriTimeseries_tensor=torch.stack(communityTimeseries)
        fmriTimeseries_norm_tensor = (fmriTimeseries_tensor - fmriTimeseries_tensor.min()) / (fmriTimeseries_tensor.max() - fmriTimeseries_tensor.min())
        community_tensor.append(fmriTimeseries_norm_tensor)
        node_idx_list.append(node_idx_list)
    normalised_communities= torch.stack(community_tensor)
    return normalised_communities,node_idx_list

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