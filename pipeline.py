from preprocess import load_graph,create_npy,build_graph
from modules import meso,local_enc,global_encoding
import os
import pickle
import torch_geometric.utils as utils
import numpy as np



def preprocess(files, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, file in enumerate(files, start=1):
        print(f"Processing {i}/{len(files)}: {file}")

        graph = build_graph.build_graph(file)
        graph = build_graph.drop_fraction_of_neighbors(graph)
        graph = build_graph.changes_weight(graph, metric=None)

        filename = os.path.basename(file)
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{stem}.gpickle")

        with open(output_path, "wb") as f:
            pickle.dump(graph, f)

def stuct_embed(timeseries,graph,dim,gamma,fmriTimeseries,community_size,use_edge_attr=True,normalization=None):

    data=utils.from_networkx(graph)

    data.edge_attr = data.weight
    ## adding the global embedding 
    encoder = global_encoding.GlobalEncoding(
        dim = data.num_nodes,
        use_edge_attr=use_edge_attr,
        normalization=normalization
    )
    glob_embedding = encoder.eigen_vecs(data,gamma=1.0)

    ## adding the local embedding

    curv_dict = local_enc.computing_curvature(graph)
    graph = local_enc.addin_attr(graph, curv_dict)
    local_feat, weight_mat = local_enc.feature_vec(graph)

    communities, _ = meso.createLouvainCommunity(graph, community_size)
    normalised_communities, node_idx_list = meso.normaliseTimeSeries(
        communities, fmriTimeseries
    )

    node_map=meso.mesoEmbed(normalised_communities,node_idx_list)

    return local_feat,glob_embedding,node_map

if __name__=='__main__':

    DATA_DIR = "/home/snottyunhunter/ReBrainFormer/test_data/ABIDE_pcp/cpac/nofilt_noglobal"
    TARGET_DIR ="test_data_npy"
    OUTPUT_DIR="preprocessed_graphs"

    a= create_npy._prepare_raw_data(TARGET_DIR)

