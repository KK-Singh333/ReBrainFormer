import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pyg_utils

from modules import global_encoding, local_enc, meso
from preprocess import build_graph, create_npy


PROJECT_ROOT = Path(__file__).resolve().parent


def _spectral_entropy_tensor(adjacency, gamma):
    """Spectral entropy for one adjacency or a batch of adjacencies."""
    laplacian = torch.diag_embed(adjacency.sum(dim=-1)) - adjacency
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    probabilities = torch.softmax(-gamma * eigenvalues, dim=-1)
    return -torch.xlogy(probabilities, probabilities).sum(dim=-1)


def spectral_entropy(adjacency, gamma, device="cpu"):
    """Entropy of the weighted graph Laplacian, using the selected device."""
    device = torch.device(device)
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    adjacency = torch.as_tensor(adjacency, dtype=dtype, device=device)
    return float(_spectral_entropy_tensor(adjacency, gamma).item())


def node_entanglement(adjacency, gamma, device="cpu"):
    """Change in spectral entropy when each ROI is removed."""
    device = torch.device(device)
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    adjacency = torch.as_tensor(adjacency, dtype=dtype, device=device)
    n_rois = adjacency.shape[0]
    original = _spectral_entropy_tensor(adjacency, gamma)
    indices = torch.arange(n_rois, device=device).expand(n_rois, -1)
    keep = indices[~torch.eye(n_rois, dtype=torch.bool, device=device)].reshape(n_rois, -1)

    # Chunk the batched eigensolver to bound GPU memory for larger atlases.
    scores = []
    for kept_nodes in keep.split(16):
        reduced = adjacency[kept_nodes[:, :, None], kept_nodes[:, None, :]]
        scores.append((_spectral_entropy_tensor(reduced, gamma) - original).abs())
    return torch.cat(scores).to(torch.float32).cpu().numpy()


def process_subject(npy_path, graph_dir, feature_dir, gamma, resolution, device="cpu"):
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
    pyg_graph = pyg_graph.to(device)
    encoder = global_encoding.GlobalEncoding(
        dim=n_rois, use_edge_attr=True, normalization=None
    )
    global_spectral = encoder.eigen_vecs(pyg_graph, gamma=gamma).cpu().numpy()

    affinity = nx.to_numpy_array(
        affinity_graph, nodelist=range(n_rois), weight="weight"
    )
    entanglement = node_entanglement(affinity, gamma, device=device)

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
        default=PROJECT_ROOT / "test_data/ABIDE_pcp/cpac/nofilt_noglobal",
    )
    parser.add_argument("--timeseries-dir", type=Path, default=PROJECT_ROOT / "test_data_npy")
    parser.add_argument("--graph-dir", type=Path, default=PROJECT_ROOT / "preprocessed_graphs")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "processed_features")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Device for spectral calculations (default: CUDA when available)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite-timeseries", action="store_true",
        help="Re-extract scans even when .npy files already exist",
    )
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.skip_extraction and args.overwrite_timeseries:
        parser.error("--skip-extraction and --overwrite-timeseries cannot be combined")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot see a CUDA device")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Spectral calculations: {device}; curvature and communities: CPU")

    if args.skip_extraction:
        files = sorted(args.timeseries_dir.glob("*_func_preproc*.npy"))
    else:
        files = create_npy._prepare_raw_data(
            args.timeseries_dir, args.nifti_dir,
            limit=args.limit, overwrite=args.overwrite_timeseries,
        )

    if not files:
        raise FileNotFoundError("No extracted ABIDE .npy files found")

    for path in files[: args.limit]:
        process_subject(
            path, args.graph_dir, args.feature_dir,
            args.gamma, args.resolution, device,
        )


if __name__ == "__main__":
    main()
