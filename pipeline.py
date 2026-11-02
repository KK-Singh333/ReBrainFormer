import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch_geometric.utils as pyg_utils

from modules import global_encoding, local_enc, meso
from preprocess import build_graph, create_npy


def spectral_entropy(adjacency, gamma):
    """Entropy of the weighted graph Laplacian."""
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    logits = -gamma * eigenvalues
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def node_entanglement(adjacency, gamma):
    """Change in spectral entropy when each ROI is removed."""
    original = spectral_entropy(adjacency, gamma)
    scores = np.empty(adjacency.shape[0], dtype=np.float32)

    for node in range(adjacency.shape[0]):
        reduced = np.delete(np.delete(adjacency, node, axis=0), node, axis=1)
        scores[node] = abs(spectral_entropy(reduced, gamma) - original)

    return scores


def process_subject(npy_path, graph_dir, feature_dir, gamma, resolution):
    time_by_roi = np.load(npy_path, allow_pickle=False)
    if time_by_roi.ndim != 2 or not np.isfinite(time_by_roi).all():
        raise ValueError(f"Invalid [time, ROI] array: {npy_path}")

    roi_by_time = time_by_roi.T
    n_rois = roi_by_time.shape[0]
    if n_rois < 2 or time_by_roi.shape[0] < 2:
        raise ValueError(f"Too few ROIs or time points: {npy_path}")

    correlation = np.corrcoef(roi_by_time).astype(np.float32)
    if not np.isfinite(correlation).all():
        raise ValueError(f"Undefined ROI correlation: {npy_path}")
    np.fill_diagonal(correlation, 0.0)

    # Sparsify using the existing rule, then create separate positive weights:
    # distances for curvature and affinities for Louvain/Laplacian.
    sparse_graph = nx.from_numpy_array(correlation)
    sparse_graph = build_graph.drop_fraction_of_neighbors(sparse_graph)

    distance_graph = sparse_graph.copy()
    affinity_graph = sparse_graph.copy()
    for u, v in sparse_graph.edges:
        r = float(np.clip(correlation[u, v], -1.0, 1.0))
        distance_graph[u][v]["weight"] = np.sqrt(2.0 * (1.0 - r)) + 1e-6
        affinity_graph[u][v]["weight"] = (r + 1.0) / 2.0 + 1e-6

    # Local curvature features and an edge-curvature attention matrix.
    curvature = local_enc.computing_curvature(distance_graph)
    local_enc.addin_attr(distance_graph, curvature)
    local_features, _ = local_enc.feature_vec(distance_graph)

    curvature_bias = np.zeros((n_rois, n_rois), dtype=np.float32)
    for u, v, attributes in distance_graph.edges(data=True):
        value = attributes["RicciCurvature"]
        curvature_bias[u, v] = value
        curvature_bias[v, u] = value

    # Global encoding uses affinities. Node entanglement is calculated
    # separately because eigen_vecs() returns values indexed by eigenvalue.
    pyg_graph = pyg_utils.from_networkx(affinity_graph)
    pyg_graph.edge_attr = pyg_graph.weight
    encoder = global_encoding.GlobalEncoding(
        dim=n_rois, use_edge_attr=True, normalization=None
    )
    global_spectral = encoder.eigen_vecs(pyg_graph, gamma=gamma).numpy()

    affinity = nx.to_numpy_array(
        affinity_graph, nodelist=range(n_rois), weight="weight"
    )
    entanglement = node_entanglement(affinity, gamma)

    # Community matrix and the existing mesoscopic node representations.
    communities, community_matrix = meso.createLouvainCommunity(
        affinity_graph, resolution
    )
    normalized, node_indices = meso.normaliseTimeSeries(
        communities, roi_by_time
    )
    node_map = meso.mesoEmbed(normalized, node_indices)
    meso_features = np.stack([
        np.asarray(node_map[node], dtype=np.float32)
        for node in range(n_rois)
    ])

    # Existing ReBrainFormer.forward expects one [N, 4, N] tensor.
    # Its channels are H, curvature bias, destination-node entanglement,
    # and same-community bias.
    entanglement_bias = np.broadcast_to(
        entanglement[None, :], (n_rois, n_rois)
    )
    model_input = np.stack(
        (
            correlation,
            curvature_bias,
            entanglement_bias,
            community_matrix.astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)

    for name, value in {
        "local_features": local_features,
        "global_spectral": global_spectral,
        "entanglement": entanglement,
        "meso_features": meso_features,
        "model_input": model_input,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} for {npy_path}")

    graph_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    stem = npy_path.stem

    with (graph_dir / f"{stem}.gpickle").open("wb") as stream:
        pickle.dump(affinity_graph, stream)

    np.savez_compressed(
        feature_dir / f"{stem}.npz",
        model_input=model_input,
        local_features=local_features,
        global_spectral=global_spectral,
        entanglement=entanglement,
        meso_features=meso_features,
        correlation=correlation,
    )
    print(f"Saved {stem}: {n_rois} ROIs, input {model_input.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nifti-dir",
        type=Path,
        default=Path("test_data/ABIDE_pcp/cpac/nofilt_noglobal"),
    )
    parser.add_argument("--timeseries-dir", type=Path, default=Path("test_data_npy"))
    parser.add_argument("--graph-dir", type=Path, default=Path("preprocessed_graphs"))
    parser.add_argument("--feature-dir", type=Path, default=Path("processed_features"))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    if args.skip_extraction:
        files = sorted(args.timeseries_dir.glob("*_func_preproc*.npy"))
    else:
        files = create_npy._prepare_raw_data(
            args.timeseries_dir, args.nifti_dir
        )

    if not files:
        raise FileNotFoundError("No extracted ABIDE .npy files found")

    for path in files[: args.limit]:
        process_subject(
            path, args.graph_dir, args.feature_dir,
            args.gamma, args.resolution,
        )


if __name__ == "__main__":
    main()