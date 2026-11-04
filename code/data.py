import os
import random
import hashlib

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset


def crop_to_multiple(mat, factor=4):
    H, W = mat.shape
    new_H = (H // factor) * factor
    new_W = (W // factor) * factor
    if new_H == 0 or new_W == 0:
        raise ValueError(f"Matrix too small: {mat.shape}")
    return mat[:new_H, :new_W]


def _hash_strings(values):
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_tensor(tensor):
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def preprocess_hic_matrix(mat):
    mat = crop_to_multiple(mat, factor=4).astype(np.float32)
    mat = np.log1p(mat)
    hi = np.percentile(mat, 99)
    mat = np.clip(mat, 0, hi)
    mat = mat / (mat.max() + 1e-6)
    return mat


class MultiOmicsDataset(Dataset):
    def __init__(
        self,
        hic_path,
        rna_path,
        min_cells_ratio,
        n_genes,
        mode="multi",
    ):
        self.mode = mode

        if self.mode != "rna_only":
            hic_data = pd.read_pickle(hic_path)
            raw_hic_data = []
            for cell in hic_data:
                chr_list = []
                for mat in cell:
                    proc = preprocess_hic_matrix(mat)
                    chr_list.append(proc)
                raw_hic_data.append(chr_list)

            self.n_chr = len(raw_hic_data[0])

            self.hic_data = [
                [torch.tensor(mat, dtype=torch.float32).unsqueeze(0) for mat in cell]
                for cell in raw_hic_data
            ]
        else:
            self.hic_data = None
            self.n_chr = 0

        gene_data = pd.read_csv(rna_path, index_col=0, sep=None, engine="python").T
        adata = sc.AnnData(gene_data)
        sc.pp.filter_genes(adata, min_cells=1)

        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)
        sc.pp.highly_variable_genes(
            adata_norm,
            n_top_genes=n_genes,
            flavor="seurat",
        )

        var_info = adata_norm.var.copy()
        var_info["gene_name"] = var_info.index.astype(str)
        var_info["orig_order"] = np.arange(var_info.shape[0])

        if "highly_variable_rank" in var_info.columns:
            selected_genes = (
                var_info.sort_values(
                    ["highly_variable_rank", "gene_name"],
                    ascending=[True, True],
                    kind="mergesort",
                )
                .head(n_genes)["gene_name"]
                .tolist()
            )
        else:
            selected_genes = (
                var_info.loc[var_info["highly_variable"].fillna(False)]
                .sort_values(["gene_name"], kind="mergesort")["gene_name"]
                .tolist()
            )
        if len(selected_genes) != n_genes:
            raise ValueError(
                f"Expected {n_genes} highly variable genes, got {len(selected_genes)}"
            )

        adata = adata[:, selected_genes].copy()
        self.cell_names = adata.obs_names.astype(str).tolist()
        self.gene_names = adata.var_names.astype(str).tolist()
        self.gene_data = torch.tensor(adata.X, dtype=torch.float32)
        self.gene_num = adata.shape[1]
        self.cell_names_sha256 = _hash_strings(self.cell_names)
        self.gene_names_sha256 = _hash_strings(self.gene_names)
        self.gene_data_sha256 = _hash_tensor(self.gene_data)

        if self.mode != "rna_only":
            assert len(self.hic_data) == self.gene_data.shape[0]

    def __len__(self):
        return len(self.gene_data)

    def __getitem__(self, idx):
        return {
            "hic": None if self.mode == "rna_only" else self.hic_data[idx],
            "rna": self.gene_data[idx],
        }

    def fingerprint(self):
        info = {
            "n_cells": int(self.gene_data.shape[0]),
            "n_genes": int(self.gene_data.shape[1]),
            "cell_names_sha256": self.cell_names_sha256,
            "gene_names_sha256": self.gene_names_sha256,
            "gene_data_sha256": self.gene_data_sha256,
            "first_cell_names": self.cell_names[:10],
            "first_gene_names": self.gene_names[:10],
            "rna_sum": float(self.gene_data.double().sum().item()),
            "first_rna_sum": float(self.gene_data[0].double().sum().item()),
            "first_rna_first10": [
                float(x) for x in self.gene_data[0].reshape(-1)[:10].tolist()
            ],
        }
        if self.mode != "rna_only":
            info["n_chr"] = int(self.n_chr)
            info["first_hic_sums"] = [
                float(x.double().sum().item()) for x in self.hic_data[0]
            ]
        return info


def _make_collate_fn(mode):
    def collate_fn(batch):
        out = {}
        if mode == "hic_only":
            out["rna"] = None
        else:
            out["rna"] = torch.stack([item["rna"] for item in batch])

        if mode == "rna_only":
            out["hic"] = None
        else:
            hic_list = [item["hic"] for item in batch]
            hic_list = list(zip(*hic_list))
            hic_list = [torch.stack(x) for x in hic_list]
            out["hic"] = hic_list

        return out

    return collate_fn


def create_dataloader_from_dataset(dataset, batch_size, shuffle=True, mode="multi"):
    g = torch.Generator()
    g.manual_seed(0)

    def _seed_worker(worker_id):
        np.random.seed(0 + worker_id)
        random.seed(0 + worker_id)
        torch.manual_seed(0 + worker_id)
    num_workers=4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_make_collate_fn(mode),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        generator=g,
        worker_init_fn=_seed_worker,
    )


def create_multiomics_dataloader(
    hic_path,
    rna_path,
    batch_size,
    n_genes,
    shuffle=True,
    min_cells_ratio=0.1,
    mode="multi",
):
    dataset = MultiOmicsDataset(
        hic_path=hic_path,
        rna_path=rna_path,
        min_cells_ratio=min_cells_ratio,
        n_genes=n_genes,
        mode=mode,
    )
    dataloader = create_dataloader_from_dataset(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        mode=mode,
    )
    return dataloader, dataset
