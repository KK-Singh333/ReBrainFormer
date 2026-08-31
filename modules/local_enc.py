import pickle
import networkx as nx
import numpy as np
from GraphRicciCurvature import OllivierRicci as oc


def computing_curvature(graph):
    return oc._compute_ricci_curvature_edges(graph, weight="weight", edge_list=graph.edges())


def addin_attr(graph, curv_dict):
    nx.set_edge_attributes(graph, curv_dict, "RicciCurvature")
    return graph


def feature_vec(graph):
    """cv = [mean_c, min_c, var_c, frac_neg, mrw] per eq. 5 -> feeds MLP(cv) in eq. 6"""
    nodes = list(graph.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    feats = np.zeros((len(nodes), 5))

    for n in nodes:
        i = node_idx[n]
        neighbors = list(graph.neighbors(n))
        if len(neighbors) == 0:
            feats[i] = [0.0, 0.0, 0.0, 0.0, 0.0]
            continue

        incident = np.array([graph[n][nbr]["RicciCurvature"] for nbr in neighbors])
        mean_c = incident.mean()
        min_c = incident.min()
        var_c = incident.var()
        frac_neg = (incident < 0).mean()

        weights = np.array([graph[n][nbr].get("weight", 1.0) for nbr in neighbors])
        w_sum = weights.sum()
        mrw = (weights * incident).sum() / w_sum if w_sum > 0 else 0.0

        feats[i] = [mean_c, min_c, var_c, frac_neg, mrw]

    return feats



if __name__ == "__main__":
    with open("graph.gpickle", "rb") as f:
        graph = pickle.load(f)

    curv_dict = computing_curvature(graph)
    graph = addin_attr(graph, curv_dict)
    cv = feature_vec(graph)             # (N,5) -> CurvatureMLP -> e_curv_v, additive into h^(0)
