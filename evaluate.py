import os
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_z_performance(
    model,
    dataloader_eval,
    true_labels_df,
    device,
    epoch,
    save_root,
    cal,
    n_cluster,
    reps=("mu",),
    save_mu_path=None,
):
    model.eval()
    os.makedirs(save_root, exist_ok=True)


    all_latent_dict = {rep: [] for rep in reps}

    with torch.no_grad():
        for data in dataloader_eval:
            if model.mode == "hic_only":
                rna = None
            else:
                rna = data["rna"].to(device)

            if model.mode == "rna_only":
                hic = None
            else:
                hic = [x.to(device) for x in data["hic"]]

            if model.mode == "hic_only":
                mu, logvar, z = model.combineEncoder(rna_x=None, hic_x=hic, mode="hic_only")
                rep_map = {"mu": mu, "z": z}
            elif model.mode == "rna_only":
                mu, logvar, z = model.combineEncoder(rna_x=rna, hic_x=None, mode="rna_only")
                rep_map = {"mu": mu, "z": z}
            else:
                rep_map = model.get_all_representations(rna, hic)

            for rep in reps:
                if rep not in rep_map:
                    continue
                latent = rep_map[rep]
                all_latent_dict[rep].append(latent.detach().cpu().numpy())

    results = defaultdict(dict)

    for rep in list(all_latent_dict.keys()):
        if len(all_latent_dict[rep]) == 0:
            continue

        latent_representations = np.concatenate(all_latent_dict[rep], axis=0)
        if not np.isfinite(latent_representations).all():
            raise ValueError(f"Latent contains NaN/Inf for rep={rep}")

        if rep == "mu" and save_mu_path is not None:
            np.save(save_mu_path, latent_representations)
            print(f"Saved embedding_mu to: {save_mu_path}")

        sample_names = np.array([str(i) for i in range(latent_representations.shape[0])])
        for label_name in true_labels_df.columns:
            labels = true_labels_df[label_name].copy()
            labels.index = labels.index.astype(str)
            gt = labels.loc[sample_names].values

            pred_kmeans = KMeans(
                n_clusters=n_cluster[label_name],
                random_state=42,
                n_init=20,
            ).fit_predict(latent_representations)

            if cal:
                ari_kmeans = adjusted_rand_score(gt, pred_kmeans)
                nmi_kmeans = normalized_mutual_info_score(gt, pred_kmeans)
                results[label_name][f"ari_kmeans_{rep}"] = ari_kmeans
                results[label_name][f"nmi_kmeans_{rep}"] = nmi_kmeans


    model.train()
    return results
    


def _move_batch_for_inference(data, model, device):
    rna = None if model.mode == "hic_only" else data["rna"].to(device)
    hic = None if model.mode == "rna_only" else [x.to(device) for x in data["hic"]]
    return rna, hic


def _decode_outputs(model, rna, hic):
    if model.mode == "hic_only":
        mu, logvar, z = model.combineEncoder(rna_x=None, hic_x=hic, mode="hic_only")
        decode_z = mu
        pred_hic = []
        if model.combineDecoder.n_chr > 0:
            z_expand = decode_z.unsqueeze(1).repeat(1, model.combineDecoder.n_chr, 1)
            for i, decoder in enumerate(model.combineDecoder.HiCdecoder):
                pred_hic.append(decoder(z_expand[:, i, :]))
        return None, pred_hic

    rna = torch.nan_to_num(rna, nan=0.0, posinf=0.0, neginf=0.0)
    if model.mode == "rna_only":
        mu, _, _ = model.combineEncoder(rna_x=rna, hic_x=None, mode="rna_only")
    else:
        mu, _, _ = model.combineEncoder(rna_x=rna, hic_x=hic, mode="multi")

    decode_z = mu
    _, _, log_l = model.libraryEncoder(rna)
    _, _, pred_rna_rate, pred_hic = model.combineDecoder(
        decode_z,
        log_l,
        decode_hic=(model.mode != "rna_only"),
    )
    return pred_rna_rate, pred_hic


def export_model_outputs(
    model,
    dataloader_eval,
    device,
    save_root,
    epoch,
    model_config=None,
    dataset=None,
    save_outputs=True,
):
    """
    Save the trained model, latent representations, and decoded outputs.

    Saved files:
        model_full_epoch{epoch}.pt
        model_checkpoint_epoch{epoch}.pt
        model_config_epoch{epoch}.json
        embedding_{rep}_epoch{epoch}.npy
        embedding_{rep}_epoch{epoch}.csv
        input_encoder_scrna_epoch{epoch}.npy
        input_encoder_scrna_genes_epoch{epoch}.txt
        input_encoder_schic_epoch{epoch}.npz
        input_encoder_schic_chr{idx}_epoch{epoch}.npy
        output_scrna_epoch{epoch}.npy
        output_schic_epoch{epoch}.npz
        output_schic_chr{idx}_epoch{epoch}.npy
    """
    os.makedirs(save_root, exist_ok=True)
    model.eval()

    full_model_path = os.path.join(save_root, f"model_full_epoch{epoch}.pt")
    checkpoint_path = os.path.join(save_root, f"model_checkpoint_epoch{epoch}.pt")
    config_path = os.path.join(save_root, f"model_config_epoch{epoch}.json")

    config_to_save = dict(model_config or {})
    if dataset is not None:
        config_to_save["dataset_fingerprint"] = (
            dataset.fingerprint() if hasattr(dataset, "fingerprint") else None
        )

    torch.save(model, full_model_path)
    torch.save(
        {
            "epoch": epoch,
            "mode": model.mode,
            "model_state_dict": model.state_dict(),
            "model_config": config_to_save,
            "z_dim": getattr(model, "z_dim", None),
            "hic_weight": getattr(model, "hic_weight", None),
            "n_genes": getattr(model, "n_genes", None),
        },
        checkpoint_path,
    )
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2, sort_keys=True)
    print(f"Saved full model to: {full_model_path}")
    print(f"Saved model checkpoint to: {checkpoint_path}")
    print(f"Saved model config to: {config_path}")

    rep_arrays = defaultdict(list)
    input_rna_batches = []
    input_hic_batches = None
    pred_rna_batches = []
    pred_hic_batches = None

    with torch.no_grad():
        for data in dataloader_eval:
            rna, hic = _move_batch_for_inference(data, model, device)
            if rna is not None:
                input_rna_batches.append(rna.detach().cpu().numpy())
            if hic is not None:
                if input_hic_batches is None:
                    input_hic_batches = [[] for _ in range(len(hic))]
                for chr_idx, chr_input in enumerate(hic):
                    input_hic_batches[chr_idx].append(
                        chr_input.detach().cpu().numpy().squeeze(1)
                    )

            reps = model.get_all_representations(rna_x=rna, hic_x=hic)
            for rep_name, rep_value in reps.items():
                rep_arrays[rep_name].append(rep_value.detach().cpu().numpy())

            if save_outputs:
                pred_rna_rate, pred_hic = _decode_outputs(model, rna, hic)
                if pred_rna_rate is not None:
                    pred_rna_batches.append(pred_rna_rate.detach().cpu().numpy())
                if pred_hic:
                    if pred_hic_batches is None:
                        pred_hic_batches = [[] for _ in range(len(pred_hic))]
                    for chr_idx, chr_pred in enumerate(pred_hic):
                        pred_hic_batches[chr_idx].append(
                            chr_pred.detach().cpu().numpy().squeeze(1)
                        )

    for rep_name, arrays in rep_arrays.items():
        if not arrays:
            continue
        arr = np.concatenate(arrays, axis=0)
        if not np.isfinite(arr).all():
            raise ValueError(f"Latent contains NaN/Inf for rep={rep_name}")
        rep_path = os.path.join(save_root, f"embedding_{rep_name}_epoch{epoch}.npy")
        np.save(rep_path, arr)
        print(f"Saved embedding_{rep_name} to: {rep_path}")
        rep_csv_path = os.path.join(save_root, f"embedding_{rep_name}_epoch{epoch}.csv")
        rep_df = pd.DataFrame(
            arr,
            columns=[f"{rep_name}_{i}" for i in range(arr.shape[1])],
        )
        rep_df.to_csv(rep_csv_path, index=False)
        print(f"Saved embedding_{rep_name} CSV to: {rep_csv_path}")

    if input_rna_batches:
        input_rna = np.concatenate(input_rna_batches, axis=0)
        input_rna_path = os.path.join(save_root, f"input_encoder_scrna_epoch{epoch}.npy")
        np.save(input_rna_path, input_rna)
        print(f"Saved encoder input scRNA to: {input_rna_path}")
        gene_names = getattr(dataset, "gene_names", None) if dataset is not None else None
        if gene_names is not None:
            gene_path = os.path.join(
                save_root,
                f"input_encoder_scrna_genes_epoch{epoch}.txt",
            )
            with open(gene_path, "w") as f:
                for gene_name in gene_names:
                    f.write(f"{gene_name}\n")
            print(f"Saved encoder input scRNA genes to: {gene_path}")

    if input_hic_batches:
        input_hic_outputs = {}
        for chr_idx, batches in enumerate(input_hic_batches):
            arr = np.concatenate(batches, axis=0)
            key = f"chr_{chr_idx}"
            input_hic_outputs[key] = arr
            chr_path = os.path.join(
                save_root,
                f"input_encoder_schic_chr{chr_idx}_epoch{epoch}.npy",
            )
            np.save(chr_path, arr)
            print(f"Saved encoder input scHiC chr{chr_idx} to: {chr_path}")
        hic_npz_path = os.path.join(save_root, f"input_encoder_schic_epoch{epoch}.npz")
        np.savez_compressed(hic_npz_path, **input_hic_outputs)
        print(f"Saved encoder input scHiC archive to: {hic_npz_path}")

    if save_outputs and pred_rna_batches:
        pred_rna = np.concatenate(pred_rna_batches, axis=0)
        rna_path = os.path.join(save_root, f"output_scrna_epoch{epoch}.npy")
        np.save(rna_path, pred_rna)
        print(f"Saved output scRNA to: {rna_path}")

    if save_outputs and pred_hic_batches:
        hic_outputs = {}
        for chr_idx, batches in enumerate(pred_hic_batches):
            arr = np.concatenate(batches, axis=0)
            key = f"chr_{chr_idx}"
            hic_outputs[key] = arr
            chr_path = os.path.join(save_root, f"output_schic_chr{chr_idx}_epoch{epoch}.npy")
            np.save(chr_path, arr)
            print(f"Saved output scHiC chr{chr_idx} to: {chr_path}")

        hic_npz_path = os.path.join(save_root, f"output_schic_epoch{epoch}.npz")
        np.savez_compressed(hic_npz_path, **hic_outputs)
        print(f"Saved output scHiC archive to: {hic_npz_path}")

    manifest = {
        "epoch": epoch,
        "files": {
            "full_model": os.path.basename(full_model_path),
            "checkpoint": os.path.basename(checkpoint_path),
            "model_config": os.path.basename(config_path),
            "final_eval_metrics": "final_eval_metrics.csv",
            "train_history": f"train_history_seed{config_to_save.get('seed', 'unknown')}.csv",
            "run_fingerprint": "run_fingerprint.json",
            "first_batch_fingerprint": (
                f"first_batch_fingerprint_seed{config_to_save.get('seed', 'unknown')}.json"
            ),
        },
        "notes": (
            "This folder is intended as a frozen result bundle. "
            "Use saved embeddings, model outputs, model checkpoint, processed inputs, "
            "and fingerprints for downstream/reviewer analyses without retraining."
        ),
    }
    manifest_path = os.path.join(save_root, "artifact_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved artifact manifest to: {manifest_path}")

    model.train()
    return {
        "full_model": full_model_path,
        "checkpoint": checkpoint_path,
        "model_config": config_path,
        "manifest": manifest_path,
    }

def explain_one_rep_dim(
    model,
    data,
    latent_dim,
    device,
    rep_name="mu",
):
    """
    用 get_all_representations()
    解释某个 representation dimension
    """

    model.eval()

    rna = data["rna"].to(device).clone().detach().requires_grad_(True)
    hic = [
        x.to(device).clone().detach().requires_grad_(True)
        for x in data["hic"]
    ]

    reps = model.get_all_representations(
        rna_x=rna,
        hic_x=hic
    )

    if rep_name not in reps:
        raise ValueError(
            f"{rep_name} not found. Available reps: {list(reps.keys())}"
        )

    rep = reps[rep_name]

    # 解释第 latent_dim 维
    score = rep[:, latent_dim].mean()

    model.zero_grad()
    score.backward()

    # RNA attribution
    rna_attr = (rna.grad * rna).detach().cpu()

    # HiC attribution
    hic_attr = []
    for x in hic:
        hic_attr.append(
            (x.grad * x).detach().cpu()
        )

    return rna_attr, hic_attr


def _move_batch_for_explain(data, model, device):
    if model.mode == "hic_only":
        rna = None
    else:
        rna = data["rna"].to(device).clone().detach().requires_grad_(True)

    if model.mode == "rna_only":
        hic = None
    else:
        hic = [
            x.to(device).clone().detach().requires_grad_(True)
            for x in data["hic"]
        ]

    return rna, hic


def _get_representation(model, rna, hic, rep_name):
    if model.mode == "hic_only":
        mu, logvar, z = model.combineEncoder(rna_x=None, hic_x=hic, mode="hic_only")
        rep_map = {"mu": mu, "z": z}
    elif model.mode == "rna_only":
        mu, logvar, z = model.combineEncoder(rna_x=rna, hic_x=None, mode="rna_only")
        rep_map = {"mu": mu, "z": z}
    else:
        rep_map = model.get_all_representations(rna_x=rna, hic_x=hic)

    if rep_name not in rep_map:
        raise ValueError(f"{rep_name} not found. Available reps: {list(rep_map.keys())}")
    return rep_map[rep_name]


def explain_latent_features(
    model,
    dataloader_eval,
    dataset,
    device,
    save_root,
    rep_name="mu",
    top_k=50,
):
    """
    Explain each latent feature with gradient * input attribution.

    For each latent dimension, the score is the mean latent value across all
    cells in the evaluation dataset:
        sum(rep[batch_cells, latent_dim]) / n_total_samples

    Outputs:
        latent_feature_rna_top.csv
        latent_feature_hic_top.csv
        latent_feature_modality_summary.csv
    """
    if model.mode not in {"multi", "rna_only", "hic_only"}:
        raise ValueError(f"Unknown mode: {model.mode}")

    os.makedirs(save_root, exist_ok=True)
    model.eval()

    rna_rows = []
    hic_rows = []
    summary_rows = []
    gene_names = getattr(dataset, "gene_names", None)
    n_total_samples = len(dataset)

    for latent_dim in range(model.z_dim):
        rna_scores = None
        hic_scores = None
        n_samples = 0

        for data in dataloader_eval:
            rna, hic = _move_batch_for_explain(data, model, device)
            batch_size = data["hic"][0].shape[0] if model.mode == "hic_only" else data["rna"].shape[0]
            n_samples += batch_size

            rep = _get_representation(model, rna, hic, rep_name)
            if latent_dim >= rep.shape[1]:
                raise ValueError(f"latent_dim={latent_dim} is out of range for rep={rep_name} with shape={rep.shape}")

            score = rep[:, latent_dim].sum() / float(n_total_samples)
            model.zero_grad(set_to_none=True)
            score.backward()

            if rna is not None:
                batch_rna_attr = (rna.grad * rna).detach().abs().sum(dim=0).cpu()
                rna_scores = batch_rna_attr if rna_scores is None else rna_scores + batch_rna_attr

            if hic is not None:
                batch_hic_scores = [
                    (x.grad * x).detach().abs().sum(dim=0).squeeze(0).cpu()
                    for x in hic
                ]
                if hic_scores is None:
                    hic_scores = batch_hic_scores
                else:
                    hic_scores = [
                        old + new
                        for old, new in zip(hic_scores, batch_hic_scores)
                    ]

        rna_total = float(rna_scores.sum().item()) if rna_scores is not None else 0.0
        hic_total = float(sum(x.sum().item() for x in hic_scores)) if hic_scores is not None else 0.0
        raw_total = rna_total + hic_total

        rna_num_features = int(rna_scores.numel()) if rna_scores is not None else 0
        hic_num_features = int(sum(x.numel() for x in hic_scores)) if hic_scores is not None else 0

        rna_mean_attr = rna_total / rna_num_features if rna_num_features > 0 else np.nan
        hic_mean_attr = hic_total / hic_num_features if hic_num_features > 0 else np.nan

        if rna_scores is not None:
            rna_top_values, rna_top_indices = torch.topk(
                rna_scores,
                k=min(top_k, rna_scores.numel()),
            )
            rna_topk_mean_attr = float(rna_top_values.mean().item())
        else:
            rna_top_values, rna_top_indices = None, None
            rna_topk_mean_attr = np.nan

        if hic_scores is not None:
            hic_flat_all = torch.cat([x.reshape(-1) for x in hic_scores])
            hic_top_values_all, _ = torch.topk(
                hic_flat_all,
                k=min(top_k, hic_flat_all.numel()),
            )
            hic_topk_mean_attr = float(hic_top_values_all.mean().item())
        else:
            hic_topk_mean_attr = np.nan

        summary_rows.append({
            "latent_dim": latent_dim,
            "rep": rep_name,
            "n_samples": n_samples,
            "rna_total_attr": rna_total,
            "hic_total_attr": hic_total,
            "rna_num_features": rna_num_features,
            "hic_num_features": hic_num_features,
            "rna_mean_attr": rna_mean_attr,
            "hic_mean_attr": hic_mean_attr,
            "rna_topk_mean_attr": rna_topk_mean_attr,
            "hic_topk_mean_attr": hic_topk_mean_attr,
            "rna_raw_sum_fraction": rna_total / raw_total if raw_total > 0 else np.nan,
            "hic_raw_sum_fraction": hic_total / raw_total if raw_total > 0 else np.nan,
        })

        if rna_scores is not None:
            for rank, (idx, value) in enumerate(zip(rna_top_indices.tolist(), rna_top_values.tolist()), start=1):
                rna_rows.append({
                    "latent_dim": latent_dim,
                    "rep": rep_name,
                    "rank": rank,
                    "gene_idx": idx,
                    "gene_name": gene_names[idx] if gene_names is not None else str(idx),
                    "attribution": value,
                })

        if hic_scores is not None:
            for chr_idx, chr_scores in enumerate(hic_scores):
                flat = chr_scores.reshape(-1)
                values, indices = torch.topk(flat, k=min(top_k, flat.numel()))
                width = chr_scores.shape[1]
                for rank, (flat_idx, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                    bin_i = flat_idx // width
                    bin_j = flat_idx % width
                    hic_rows.append({
                        "latent_dim": latent_dim,
                        "rep": rep_name,
                        "chr_idx": chr_idx,
                        "rank": rank,
                        "bin_i": bin_i,
                        "bin_j": bin_j,
                        "attribution": value,
                    })

    rna_path = os.path.join(save_root, f"latent_feature_{rep_name}_rna_top.csv")
    hic_path = os.path.join(save_root, f"latent_feature_{rep_name}_hic_top.csv")
    summary_path = os.path.join(save_root, f"latent_feature_{rep_name}_modality_summary.csv")

    pd.DataFrame(rna_rows).to_csv(rna_path, index=False)
    pd.DataFrame(hic_rows).to_csv(hic_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved latent RNA attribution to: {rna_path}")
    print(f"Saved latent HiC attribution to: {hic_path}")
    print(f"Saved latent modality summary to: {summary_path}")

    model.train()
    return {
        "rna_top": rna_path,
        "hic_top": hic_path,
        "summary": summary_path,
    }



def explain_latent_features_by_celltype(
    model,
    dataloader_eval,
    dataset,
    true_labels_df,
    label_col,
    device,
    save_root,
    rep_name="mu",
    top_k=50,
    latent_dims=None,
    min_cells=20,
):
    """
    Compute cell-type-specific gradient * input attribution.

    For each cell type and selected latent dimension, the score is the mean
    latent value across all cells of that cell type:
        sum(rep[batch_celltype_cells, latent_dim]) / n_celltype_total

    The function only saves top features, not the full per-cell attribution tensor.
    Hi-C top contacts are selected globally across all chromosomes for fairness.

    Outputs:
        latent_feature_{rep_name}_celltype_rna_top.csv
        latent_feature_{rep_name}_celltype_hic_top.csv
        latent_feature_{rep_name}_celltype_modality_summary.csv
    """
    if model.mode not in {"multi", "rna_only", "hic_only"}:
        raise ValueError(f"Unknown mode: {model.mode}")
    if true_labels_df is None:
        raise ValueError("true_labels_df is required for cell-type-specific attribution")
    if label_col not in true_labels_df.columns:
        raise ValueError(
            f"label_col={label_col!r} not found. Available columns: {list(true_labels_df.columns)}"
        )

    os.makedirs(save_root, exist_ok=True)
    model.eval()

    labels = true_labels_df[label_col].astype(str).reset_index(drop=True)
    celltype_counts = labels.value_counts()
    celltypes = celltype_counts[celltype_counts >= min_cells].index.tolist()
    if len(celltypes) == 0:
        raise ValueError(f"No cell types have at least min_cells={min_cells} cells")

    if latent_dims is None:
        latent_dims = list(range(model.z_dim))
    else:
        latent_dims = [int(x) for x in latent_dims]

    gene_names = getattr(dataset, "gene_names", None)
    rna_rows = []
    hic_rows = []
    summary_rows = []

    print(
        "Running cell-type-specific latent attribution:",
        f"label_col={label_col}",
        f"n_celltypes={len(celltypes)}",
        f"n_latent_dims={len(latent_dims)}",
        f"top_k={top_k}",
    )

    total_pairs = len(celltypes) * len(latent_dims)
    completed_pairs = 0
    start_time = time.time()

    for celltype_idx, celltype in enumerate(celltypes, start=1):
        cell_mask_all = (labels == celltype).to_numpy()
        n_celltype = int(cell_mask_all.sum())

        print(
            f"[celltype {celltype_idx}/{len(celltypes)}] "
            f"{celltype} n_cells={n_celltype}",
            flush=True,
        )

        for dim_idx, latent_dim in enumerate(latent_dims, start=1):
            pair_start_time = time.time()
            print(
                f"  start pair {completed_pairs + 1}/{total_pairs}: "
                f"celltype={celltype!r} dim={latent_dim} "
                f"({dim_idx}/{len(latent_dims)} dims for this celltype)",
                flush=True,
            )

            rna_scores = None
            hic_scores = None
            n_samples = 0
            offset = 0

            for data in dataloader_eval:
                rna, hic = _move_batch_for_explain(data, model, device)
                batch_size = data["hic"][0].shape[0] if model.mode == "hic_only" else data["rna"].shape[0]

                batch_mask_np = cell_mask_all[offset: offset + batch_size]
                offset += batch_size
                if not batch_mask_np.any():
                    continue

                batch_mask = torch.tensor(batch_mask_np, dtype=torch.bool, device=device)
                n_samples += int(batch_mask.sum().item())

                rep = _get_representation(model, rna, hic, rep_name)
                if latent_dim >= rep.shape[1]:
                    raise ValueError(
                        f"latent_dim={latent_dim} is out of range for rep={rep_name} with shape={rep.shape}"
                    )

                score = rep[batch_mask, latent_dim].sum() / float(n_celltype)
                model.zero_grad(set_to_none=True)
                score.backward()

                if rna is not None:
                    batch_rna_attr = (
                        rna.grad[batch_mask] * rna[batch_mask]
                    ).detach().abs().sum(dim=0).cpu()
                    rna_scores = batch_rna_attr if rna_scores is None else rna_scores + batch_rna_attr

                if hic is not None:
                    batch_hic_scores = []
                    for x in hic:
                        attr = (
                            x.grad[batch_mask] * x[batch_mask]
                        ).detach().abs().sum(dim=0).squeeze(0).cpu()
                        batch_hic_scores.append(attr)

                    if hic_scores is None:
                        hic_scores = batch_hic_scores
                    else:
                        hic_scores = [
                            old + new
                            for old, new in zip(hic_scores, batch_hic_scores)
                        ]

            if n_samples == 0:
                continue

            rna_total = float(rna_scores.sum().item()) if rna_scores is not None else 0.0
            hic_total = float(sum(x.sum().item() for x in hic_scores)) if hic_scores is not None else 0.0
            raw_total = rna_total + hic_total

            rna_num_features = int(rna_scores.numel()) if rna_scores is not None else 0
            hic_num_features = int(sum(x.numel() for x in hic_scores)) if hic_scores is not None else 0
            rna_mean_attr = rna_total / rna_num_features if rna_num_features > 0 else np.nan
            hic_mean_attr = hic_total / hic_num_features if hic_num_features > 0 else np.nan

            if rna_scores is not None:
                rna_top_values, rna_top_indices = torch.topk(
                    rna_scores,
                    k=min(top_k, rna_scores.numel()),
                )
                rna_topk_mean_attr = float(rna_top_values.mean().item())
                rna_top_sum = float(rna_top_values.sum().item())
            else:
                rna_top_values, rna_top_indices = None, None
                rna_topk_mean_attr = np.nan
                rna_top_sum = 0.0

            if hic_scores is not None:
                hic_flat_all = torch.cat([x.reshape(-1) for x in hic_scores])
                hic_top_values_all, hic_top_indices_all = torch.topk(
                    hic_flat_all,
                    k=min(top_k, hic_flat_all.numel()),
                )
                hic_topk_mean_attr = float(hic_top_values_all.mean().item())
                hic_top_sum = float(hic_top_values_all.sum().item())
            else:
                hic_top_values_all, hic_top_indices_all = None, None
                hic_topk_mean_attr = np.nan
                hic_top_sum = 0.0

            top_total = rna_top_sum + hic_top_sum
            summary_rows.append({
                "celltype": celltype,
                "latent_dim": latent_dim,
                "rep": rep_name,
                "n_samples": n_samples,
                "n_celltype_total": n_celltype,
                "rna_total_attr": rna_total,
                "hic_total_attr": hic_total,
                "rna_num_features": rna_num_features,
                "hic_num_features": hic_num_features,
                "rna_mean_attr": rna_mean_attr,
                "hic_mean_attr": hic_mean_attr,
                "rna_topk_mean_attr": rna_topk_mean_attr,
                "hic_topk_mean_attr": hic_topk_mean_attr,
                "rna_raw_sum_fraction": rna_total / raw_total if raw_total > 0 else np.nan,
                "hic_raw_sum_fraction": hic_total / raw_total if raw_total > 0 else np.nan,
                "rna_topk_fraction": rna_top_sum / top_total if top_total > 0 else np.nan,
                "hic_topk_fraction": hic_top_sum / top_total if top_total > 0 else np.nan,
            })

            if rna_scores is not None:
                for rank, (idx, value) in enumerate(
                    zip(rna_top_indices.tolist(), rna_top_values.tolist()),
                    start=1,
                ):
                    rna_rows.append({
                        "celltype": celltype,
                        "latent_dim": latent_dim,
                        "rep": rep_name,
                        "n_samples": n_samples,
                        "rank": rank,
                        "gene_idx": idx,
                        "gene_name": gene_names[idx] if gene_names is not None else str(idx),
                        "attribution": value,
                    })

            if hic_scores is not None:
                chr_offsets = []
                running = 0
                for chr_scores in hic_scores:
                    chr_offsets.append(running)
                    running += chr_scores.numel()

                chr_rank_counter = {i: 0 for i in range(len(hic_scores))}
                for global_rank, (flat_global_idx, value) in enumerate(
                    zip(hic_top_indices_all.tolist(), hic_top_values_all.tolist()),
                    start=1,
                ):
                    chr_idx = max(
                        i for i, start in enumerate(chr_offsets)
                        if start <= flat_global_idx
                    )
                    local_idx = flat_global_idx - chr_offsets[chr_idx]
                    chr_scores = hic_scores[chr_idx]
                    width = chr_scores.shape[1]
                    bin_i = local_idx // width
                    bin_j = local_idx % width
                    chr_rank_counter[chr_idx] += 1

                    hic_rows.append({
                        "celltype": celltype,
                        "latent_dim": latent_dim,
                        "rep": rep_name,
                        "n_samples": n_samples,
                        "chr_idx": chr_idx,
                        "global_rank": global_rank,
                        "chr_rank": chr_rank_counter[chr_idx],
                        "bin_i": int(bin_i),
                        "bin_j": int(bin_j),
                        "attribution": value,
                    })

            completed_pairs += 1
            elapsed = time.time() - start_time
            pair_elapsed = time.time() - pair_start_time
            avg_per_pair = elapsed / completed_pairs if completed_pairs else np.nan
            remaining_pairs = total_pairs - completed_pairs
            eta_seconds = avg_per_pair * remaining_pairs if completed_pairs else np.nan
            print(
                f"  done pair {completed_pairs}/{total_pairs}: "
                f"celltype={celltype!r} dim={latent_dim} "
                f"n_samples={n_samples} "
                f"pair_time={pair_elapsed / 60:.1f} min "
                f"elapsed={elapsed / 60:.1f} min "
                f"eta={eta_seconds / 60:.1f} min",
                flush=True,
            )

    rna_path = os.path.join(save_root, f"latent_feature_{rep_name}_celltype_rna_top.csv")
    hic_path = os.path.join(save_root, f"latent_feature_{rep_name}_celltype_hic_top.csv")
    summary_path = os.path.join(save_root, f"latent_feature_{rep_name}_celltype_modality_summary.csv")

    pd.DataFrame(rna_rows).to_csv(rna_path, index=False)
    pd.DataFrame(hic_rows).to_csv(hic_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved celltype RNA attribution to: {rna_path}")
    print(f"Saved celltype HiC attribution to: {hic_path}")
    print(f"Saved celltype modality summary to: {summary_path}")

    model.train()
    return {
        "rna_top": rna_path,
        "hic_top": hic_path,
        "summary": summary_path,
    }
