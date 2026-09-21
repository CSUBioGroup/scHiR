import itertools
import json
import os
import platform

import torch

from data import create_dataloader_from_dataset, create_multiomics_dataloader

from models import VAE
from train import train_multiomics

from utils import set_seed


DEFAULT_PARAM_GRID = {
    "lr": [5e-4],
    "batch_size": [32],
    "z_dim": [32],
    "hic_weight": [1e-3],
    "n_genes": [2000],
    "rna_fc_layers": [3],
    "seed": [0,2025],
    "max_beta": [0.05],
    "hic_preprocess": ["simple"],
    "mode": ["rna_only", "hic_only"],
    #"multi",
}
# 

DEFAULT_STABLE_VARIANTS = [
    {
        "anti_collapse_weight": 3.0,
        "anti_collapse_target_r": 0.75,
        "anti_collapse_start_epoch": 1,
    },
]

DEFAULT_FIXED_PARAMS = {
    "hic_hid_dim": 200,
    "rna_hid_dim": 128,
    "latent_dim": 128,
    "save_artifacts": False,
    "cal": True,
    "auto_balance_hic_weight": True,
    "auto_balance_start_epoch": 5,
    "auto_balance_window": 5,
    "hic_weight_min_factor": 0.001,
    "hic_weight_max_factor": 1000.0,
    "anti_collapse_weight": 3.0,
    "anti_collapse_target_r": 0.75,
    "anti_collapse_start_epoch": 1,
    "vmf_sampling_warmup_epochs": 10,
    "hic_resolution": "1mb",
    "device": "cuda:0",
}
def demo(
    hic_path,
    rna_path,
    true_labels,
    hic_hid_dim,
    rna_fc_layers,
    rna_hid_dim,
    z_dim,
    latent_dim,
    batch_size,
    epochs,
    lr,
    hic_resolution,
    hic_weight,
    save_root,
    n_genes,
    max_beta,
    seed,
    hic_preprocess,
    device="cuda:0",
    cal=True,
    n_cluster=None,
    mode="multi",
    auto_balance_hic_weight=True,
    auto_balance_start_epoch=5,
    auto_balance_window=5,
    hic_weight_min_factor=0.001,
    hic_weight_max_factor=1000.0,
    anti_collapse_weight=3.0,
    anti_collapse_target_r=0.75,
    anti_collapse_start_epoch=1,
    vmf_sampling_warmup_epochs=10,
    output_dir=None,
    save_artifacts=False,
    explain=False,
    explain_top_k=50,
    explain_rep="mu",
):
    os.makedirs(save_root, exist_ok=True)

    set_seed(seed)
    torch.manual_seed(seed)

    print("创建数据加载器")
    dataloader, dataset = create_multiomics_dataloader(
        hic_path,
        rna_path,
        batch_size,
        n_genes,
        shuffle=True,
        seed=seed,
        hic_preprocess=hic_preprocess,
        mode=mode,
    )
    print("基因数量：", dataset.gene_num)
    fingerprint = {
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "dataset": dataset.fingerprint(),
    }
    fingerprint_path = os.path.join(save_root, "run_fingerprint.json")
    with open(fingerprint_path, "w") as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)
    print("Saved run fingerprint to:", fingerprint_path)
    print("初始化模型")
    if mode == "rna_only":
        chr_input_dims = []
    else:
        sample_cell = dataset[0]["hic"]
        chr_input_dims = [m.shape[-1] for m in sample_cell]

    model = VAE(
        hic_hid_dim=hic_hid_dim,
        rna_input_dim=dataset.gene_num,
        rna_fc_layers=rna_fc_layers,
        rna_hid_dim=rna_hid_dim,
        library_n_in=dataset.gene_num,
        library_n_out=1,
        library_n_layers=1,
        library_n_hidden=128,
        library_dropout_rate=0.1,
        rna_output_dim=dataset.gene_num,
        latent_dim=latent_dim,
        z_dim=z_dim,
        hic_weight=hic_weight,
        n_genes=n_genes,
        chr_input_dims=chr_input_dims,
        mode=mode,
        anti_collapse_weight=anti_collapse_weight,
        anti_collapse_target_r=anti_collapse_target_r,
    ).to(device)

    trained_model = train_multiomics(
        model,
        dataloader,
        epochs=epochs,
        lr=lr,
        save_root=save_root,
        device=device,
        max_beta=max_beta,
        seed=seed,
        target_hic_ratio=hic_weight,
        auto_balance_after_first_epoch=auto_balance_hic_weight,
        auto_balance_start_epoch=auto_balance_start_epoch,
        auto_balance_window=auto_balance_window,
        hic_weight_min_factor=hic_weight_min_factor,
        hic_weight_max_factor=hic_weight_max_factor,
        vmf_sampling_warmup_epochs=vmf_sampling_warmup_epochs,
        anti_collapse_start_epoch=anti_collapse_start_epoch,
        mode=mode,
    )

    dataloader_eval = None
    if cal or save_artifacts or explain:
        dataloader_eval = create_dataloader_from_dataset(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            mode=mode,
        )

    if save_artifacts:
        from evaluate import export_model_outputs
        model_config = {
            "hic_path": hic_path,
            "rna_path": rna_path,
            "output_dir": output_dir,
            "n_cluster": n_cluster,
            "hic_hid_dim": hic_hid_dim,
            "rna_fc_layers": rna_fc_layers,
            "rna_hid_dim": rna_hid_dim,
            "z_dim": z_dim,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "hic_resolution": hic_resolution,
            "hic_weight": hic_weight,
            "auto_balance_hic_weight": auto_balance_hic_weight,
            "auto_balance_start_epoch": auto_balance_start_epoch,
            "auto_balance_window": auto_balance_window,
            "hic_weight_min_factor": hic_weight_min_factor,
            "hic_weight_max_factor": hic_weight_max_factor,
            "anti_collapse_weight": anti_collapse_weight,
            "anti_collapse_target_r": anti_collapse_target_r,
            "anti_collapse_start_epoch": anti_collapse_start_epoch,
            "vmf_sampling_warmup_epochs": vmf_sampling_warmup_epochs,
            "n_genes": n_genes,
            "max_beta": max_beta,
            "seed": seed,
            "hic_preprocess": hic_preprocess,
            "mode": mode,
            "gene_num": dataset.gene_num,
            "chr_input_dims": chr_input_dims,
            "cell_names_sha256": getattr(dataset, "cell_names_sha256", None),
            "gene_names_sha256": getattr(dataset, "gene_names_sha256", None),
            "gene_data_sha256": getattr(dataset, "gene_data_sha256", None),
        }
        export_model_outputs(
            model=trained_model,
            dataloader_eval=dataloader_eval,
            device=device,
            save_root=save_root,
            epoch=epochs,
            model_config=model_config,
            dataset=dataset,
            save_outputs=True,
        )

    if cal:
        from evaluate import evaluate_z_performance, explain_latent_features
        from utils import flatten_eval_results
        results = evaluate_z_performance(
            model=trained_model,
            dataloader_eval=dataloader_eval,
            true_labels_df=true_labels,
            device=device,
            epoch=epochs,
            save_root=save_root,
            cal=cal,
            n_cluster=n_cluster,
            save_mu_path=os.path.join(
                save_root,
                f"embedding_mu_epoch{epochs}.npy"
            ),
        )
        eval_csv = os.path.join(save_root, "final_eval_metrics.csv")
        torch.cuda.empty_cache()
        import pandas as pd
        pd.DataFrame([{"epoch": epochs, **flatten_eval_results(results)}]).to_csv(eval_csv, index=False)
        print(f"Saved final eval metrics to: {eval_csv}")
        manifest_path = os.path.join(save_root, "artifact_manifest.json")
        bundle_files = sorted(
            name for name in os.listdir(save_root)
            if os.path.isfile(os.path.join(save_root, name))
        )
        with open(os.path.join(save_root, "repro_bundle_files.txt"), "w") as f:
            for name in bundle_files:
                f.write(f"{name}\n")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {}
        manifest["final_eval_metrics_written"] = True
        manifest["bundle_files"] = bundle_files
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Updated artifact manifest: {manifest_path}")

    if explain:
        explain_latent_features(
            model=trained_model,
            dataloader_eval=dataloader_eval,
            dataset=dataset,
            device=device,
            save_root=save_root,
            rep_name=explain_rep,
            top_k=explain_top_k,
        )

    return trained_model


def build_save_paths(current_params):
    seed = current_params["seed"]
    batch_size = current_params["batch_size"]
    epochs = current_params["epochs"]
    lr = current_params["lr"]
    hic_resolution = current_params["hic_resolution"]
    z_dim = current_params["z_dim"]
    hic_weight = current_params["hic_weight"]
    auto_balance_hic_weight = current_params["auto_balance_hic_weight"]
    auto_balance_start_epoch = current_params["auto_balance_start_epoch"]
    auto_balance_window = current_params["auto_balance_window"]
    hic_weight_min_factor = current_params["hic_weight_min_factor"]
    hic_weight_max_factor = current_params["hic_weight_max_factor"]
    anti_collapse_weight = current_params["anti_collapse_weight"]
    anti_collapse_target_r = current_params["anti_collapse_target_r"]
    anti_collapse_start_epoch = current_params["anti_collapse_start_epoch"]
    vmf_sampling_warmup_epochs = current_params["vmf_sampling_warmup_epochs"]
    hic_hid_dim = current_params["hic_hid_dim"]
    rna_fc_layers = current_params["rna_fc_layers"]
    n_genes = current_params["n_genes"]
    max_beta = current_params["max_beta"]
    hic_preprocess = current_params["hic_preprocess"]
    mode = current_params["mode"]
    output_dir = current_params.get("output_dir", "./limcahg38")

    base = (
        f"{output_dir.rstrip('/')}/"
        f"seed{seed}_batch{batch_size}_epochs{epochs}_lr{lr}_"
        f"hicreso{hic_resolution}_zdim{z_dim}_hicratio{hic_weight}_hichiddim{hic_hid_dim}_"
        f"rnafclayers{rna_fc_layers}_gmmhvg{n_genes}_maxbeta{max_beta}_"
        f"hicbal{int(auto_balance_hic_weight)}e{auto_balance_start_epoch}w{auto_balance_window}"
        f"clip{hic_weight_min_factor}-{hic_weight_max_factor}_"
        f"collapse{anti_collapse_weight}r{anti_collapse_target_r}s{anti_collapse_start_epoch}_"
        f"vmfwarm{vmf_sampling_warmup_epochs}_"
        f"{hic_preprocess}-{mode}-poisson_vmf_spherical_stable"
    )
    return base


def hyperparameter_tuning(
    labels,
    param_grid=None,
    fixed_params=None,
    fixed_overrides=None,
    skip_existing=False,
    stable_variants=None,
):
    param_grid = param_grid or DEFAULT_PARAM_GRID
    fixed_params = fixed_params or DEFAULT_FIXED_PARAMS
    if fixed_overrides is not None:
        fixed_params = {**fixed_params, **fixed_overrides}
    fixed_params = {**fixed_params, "true_labels": labels}

    base_params = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]
    stable_variants = stable_variants or DEFAULT_STABLE_VARIANTS
    all_params = [
        {**params, **variant}
        for params in base_params
        for variant in stable_variants
    ]

    for i, params in enumerate(all_params):
        print(f"\n=== 开始实验 {i + 1}/{len(all_params)} ===")
        print("参数组合:", params)

        current_params = {**fixed_params, **params}
        save_root = build_save_paths(current_params)
        current_params["save_root"] = save_root

        eval_csv = os.path.join(save_root, "final_eval_metrics.csv")

        if skip_existing and os.path.exists(eval_csv):
            print(f"\n=== 跳过实验 {i + 1}/{len(all_params)}：final_eval_metrics.csv 已存在 ===")
            print("已存在文件:", eval_csv)
            continue

        try:
            demo(**current_params)
        except Exception as e:
            print(f"实验 {i + 1} 失败:", str(e))

    print("\n=== 所有实验完成 ===")
