import pickle
import sys
import networkx as nx
import numpy as np
import ot
from GraphRicciCurvature import OllivierRicci as oc


def computing_curvature(graph):
    if sys.platform == "win32":
        return _serial_ricci_curvature(graph)
    return oc._compute_ricci_curvature_edges(graph, weight="weight", edge_list=graph.edges())


def _serial_ricci_curvature(graph, alpha=0.5):
    """Windows-safe equivalent of the library's default OTD edge calculation.

    GraphRicciCurvature uses a Unix-only ``fork`` pool even when proc=1. For
    these undirected ROI graphs (fewer than 3000 neighbors per node), its
    OTDSinkhornMix default uses exact earth-mover distance for every edge.
    """
    if graph.is_directed():
        raise ValueError("The serial curvature path expects an undirected graph")

    nodes = list(graph.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    distances = np.asarray(nx.floyd_warshall_numpy(graph, nodelist=nodes, weight="weight"))

    def distribution(node):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return np.array([1.0]), [node]
        weights = np.exp(-np.square([graph[node][nbr]["weight"] for nbr in neighbors]))
        if weights.sum() > 1e-7:
            masses = (1.0 - alpha) * weights / weights.sum()
        else:
            masses = np.full(len(neighbors), (1.0 - alpha) / len(neighbors))
        return np.append(masses, alpha), neighbors + [node]

    densities = {node: distribution(node) for node in nodes}
    result = {}
    for source, target in graph.edges:
        source_mass, source_nodes = densities[source]
        target_mass, target_nodes = densities[target]
        cost = distances[np.ix_(
            [node_index[node] for node in source_nodes],
            [node_index[node] for node in target_nodes],
        )]
        distance = distances[node_index[source], node_index[target]]
        result[(source, target)] = (
            0.0 if distance < 1e-7
            else 1.0 - float(ot.emd2(source_mass, target_mass, cost)) / distance
        )
    return result


def addin_attr(graph, curv_dict):
    nx.set_edge_attributes(graph, curv_dict, "RicciCurvature")
    return graph


def feature_vec(graph):
    """cv = [mean_c, min_c, var_c, frac_neg, mrw] per eq. 5 -> feeds MLP(cv) in eq. 6"""
    nodes = list(graph.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    feats = np.zeros((len(nodes), 5))
    weight_mat = np.zeros((len(nodes), len(nodes)), dtype=float)
    for n in nodes:
        i = node_idx[n]
        neighbors = list(graph.neighbors(n))
        if len(neighbors) == 0:
            feats[i] = [0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        for neighbour in neighbors:
            j = node_idx[neighbour]
            weight_mat[i, j] = graph[n][neighbour].get("weight", 1.0)
        incident = np.array([graph[n][nbr]["RicciCurvature"] for nbr in neighbors])
        mean_c = incident.mean()
        min_c = incident.min()
        var_c = incident.var()
        frac_neg = (incident < 0).mean()

        weights = np.array([graph[n][nbr].get("weight", 1.0) for nbr in neighbors])
        w_sum = weights.sum()
        mrw = (weights * incident).sum() / w_sum if w_sum > 0 else 0.0

        feats[i] = [mean_c, min_c, var_c, frac_neg, mrw]

    return feats, weight_mat



if __name__ == "__main__":
    with open("graph.gpickle", "rb") as f:
        graph = pickle.load(f)

    curv_dict = computing_curvature(graph)
    graph = addin_attr(graph, curv_dict)
    cv = feature_vec(graph)             # (N,5) -> CurvatureMLP -> e_curv_v, additive into h^(0)
