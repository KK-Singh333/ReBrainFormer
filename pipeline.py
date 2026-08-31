from preprocess import load_graph,create_npy,build_graph
from modules import meso,local_enc,global_encoding
import os
import pickle
import torch_geometric.utils as utils

DATA_DIR = "/home/snottyunhunter/ReBrainFormer/test_data/ABIDE_pcp/cpac/nofilt_noglobal"
TARGET_DIR ="test_data_npy"
OUTPUT_DIR="preprocessed_graphs"

a= create_npy._prepare_raw_data(TARGET_DIR)

def preprocess(DATA_DIR,TARGET_DIR,OUTPUT_DIR):

    for i, file in enumerate(TARGET_DIR, start=1):

        print(f"Processing {i}/{len(TARGET_DIR)}: {file}")

        graph = build_graph(file)
        graph = build_graph.drop_fraction_of_neighbors(graph)
        graph = build_graph.changes_weight(graph, metric=None)

        

        output_path = os.path.join(
            OUTPUT_DIR,
            f"graph_{i}.gpickle"
        )

        with open(output_path, "wb") as f:
            pickle.dump(graph, f)


def stuct_embed(graph,dim,gamma,fmriTimeseries,community_size,use_edge_attr=True,normalization=None):

    data=utils.from_networkx(graph)
    ## adding the global embedding 
    encoder = global_encoding.GlobalEncoding(
        dim = data.num_nodes,
        use_edge_attr=use_edge_attr,
        normalization=normalization
    )
    glob_embedding = encoder.eigen_vecs(data,gamma=1.0)

    ## adding the local embedding

    local_feat=local_enc.feature_vec(graph)

    community=meso.createLouvainCommunity(graph,community_size)
    normtimeseries=meso.normaliseTimeSeries(community,fmriTimeseries)
    meso_embed=meso.mesoEmbed()

    return local_feat,glob_embedding,meso_embed









