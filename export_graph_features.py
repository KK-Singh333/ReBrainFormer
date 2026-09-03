#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
import torch_geometric.utils as utils

from configs.fmri_extractor import fMRIConfig
from modules import global_encoding, local_enc, meso
from preprocess.build_graph import build_graph, changes_weight, drop_fraction_of_neighbors
from preprocess.load_graph import _fmri_extractor, _fmri_extractor_metadata


def _ensure_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nifti_to_npy(nifti_path: Path, scratch_dir: Path) -> Path:
    """Convert a NIfTI file to a .npy array in a temporary or provided folder."""
    if nifti_path.suffix == ".npy":
        return nifti_path

    scratch_dir.mkdir(parents=True, exist_ok=True)

    event = _fmri_extractor_metadata(str(nifti_path), fMRIConfig)
    if event is None:
        raise RuntimeError(f"Failed to create metadata for {nifti_path}")

    fmri = _fmri_extractor(event)
    if fmri is None:
        raise RuntimeError(f"Failed to extract fMRI array from {nifti_path}")

    arr = fmri.numpy() if hasattr(fmri, "numpy") else np.asarray(fmri)
    out_path = scratch_dir / (nifti_path.stem + ".npy")
    np.save(out_path, arr)
    return out_path


def _dict_to_matrix(node_map):
    if not node_map:
        return np.empty((0, 0), dtype=np.float32)

    ordered_nodes = sorted(node_map.keys())
    rows = []
    for node in ordered_nodes:
        vec = _ensure_numpy_array(node_map[node]).reshape(-1)
        rows.append(vec)

    max_len = max(len(vec) for vec in rows)
    matrix = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, vec in enumerate(rows):
        matrix[i, : len(vec)] = vec.astype(np.float32)
    return matrix


def _extract_global_encoding(graph):
    data = utils.from_networkx(graph)
    encoder = global_encoding.GlobalEncoding(
        dim=data.num_nodes,
        use_edge_att=True,
        normalization=None,
    )

    global_embedding = encoder.eigen_vecs(data, gamma=1.0)
    global_embedding = _ensure_numpy_array(global_embedding).astype(np.float32)

    perturbation_result = encoder.purturbing_node(data, gamma=1.0)

    if isinstance(perturbation_result, tuple) and len(perturbation_result) == 2:
        first, second = perturbation_result
        if np.asarray(first).size == 1:
            trace_value = float(np.asarray(first).reshape(-1)[0])
            per_node_embed_diff = np.asarray(second)
        else:
            trace_value = float(np.trace(np.asarray(first)))
            per_node_embed_diff = np.asarray(second)
    else:
        per_node_embed_diff = np.asarray(perturbation_result)
        trace_value = float(np.trace(per_node_embed_diff))

    per_node_embed_diff = np.asarray(per_node_embed_diff, dtype=np.float32)
    return global_embedding, per_node_embed_diff, trace_value


def _make_graph_for_file(npy_path: Path):
    graph = build_graph(str(npy_path))
    graph = drop_fraction_of_neighbors(graph)
    graph = changes_weight(graph, metric=None)
    return graph


def _prepare_local_features(graph):
    local_curv_dict = local_enc.computing_curvature(graph)
    graph = local_enc.addin_attr(graph, local_curv_dict)
    local_embedding, weight_mat = local_enc.feature_vec(graph)
    return graph, np.asarray(local_embedding, dtype=np.float32), np.asarray(weight_mat, dtype=np.float32)


def _prepare_meso_embedding(graph, fmri_array):
    communities, _ = meso.createLouvainCommunity(graph, community_size=1)
    normalized_communities, node_idx_list = meso.normaliseTimeSeries(communities, fmri_array)
    node_map = meso.mesoEmbed(normalized_communities, node_idx_list)
    return _dict_to_matrix(node_map)


def process_file(input_path: Path, output_dir: Path, scratch_dir: Path):
    npy_path = _nifti_to_npy(input_path, scratch_dir)
    fmri_array = np.load(npy_path)
    fmri_array = np.asarray(fmri_array).squeeze()
    if fmri_array.shape[0] >= fmri_array.shape[1]:
        fmri_array = fmri_array.T

    graph = _make_graph_for_file(npy_path)
    graph, local_embedding, weight_mat = _prepare_local_features(graph)

    global_embedding, per_node_embed_diff, trace_value = _extract_global_encoding(graph)
    meso_matrix = _prepare_meso_embedding(graph, fmri_array)

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save local embedding first
    np.save(output_dir / f"{stem}_local_embedding.npy", local_embedding)
    np.save(output_dir / f"{stem}_weight_mat.npy", weight_mat)

    # Save global encoding artifacts next
    np.save(output_dir / f"{stem}_global_embedding.npy", global_embedding)
    np.save(output_dir / f"{stem}_global_per_node_embed_diff.npy", per_node_embed_diff)
    np.save(output_dir / f"{stem}_global_trace.npy", np.asarray(trace_value, dtype=np.float32))

    # Save meso embedding as a dense matrix after converting dict to matrix
    np.save(output_dir / f"{stem}_meso_embed_matrix.npy", meso_matrix)

    return {
        "local_embedding": local_embedding,
        "weight_mat": weight_mat,
        "global_embedding": global_embedding,
        "per_node_embed_diff": per_node_embed_diff,
        "global_trace": trace_value,
        "meso_embed_matrix": meso_matrix,
    }


def collect_input_files(input_dir: Path):
    input_dir = input_dir.resolve()
    files = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and (path.suffix in {".nii", ".nii.gz", ".npy"}):
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIfTI files and export graph embedding matrices as .npy files.")
    parser.add_argument("--input-dir", type=Path, default=Path("test_data"), help="Folder containing .nii, .nii.gz or .npy files.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_embeddings"), help="Folder where .npy outputs will be saved.")
    parser.add_argument("--scratch-dir", type=Path, default=Path("tmp_npy_exports"), help="Temporary folder for extracted .npy arrays.")
    args = parser.parse_args()

    files = collect_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No .nii, .nii.gz, or .npy files found in {args.input_dir}")

    for file_path in files:
        process_file(file_path, args.output_dir, args.scratch_dir)

    print(f"Processed {len(files)} file(s). Outputs saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
