import json
import os
import platform
import torch

from data import create_dataloader_from_dataset, create_multiomics_dataloader
from models import VAE
from train import train_multiomics
from utils import set_seed


# Dataset paths, output directory, resolution and epochs come from configs.py
DEFAULT_PARAMS = {
    # Model architecture and input features
    "mode": "multi",
    "n_genes": 2000,
    "hic_hid_dim": 200,
    "rna_hid_dim": 128,
    "rna_fc_layers": 3,
    "latent_dim": 128,
    "z_dim": 32,

    # Training
    "batch_size": 32,
    "lr": 5e-4,
    "max_beta": 0.05,
    "vmf_sampling_warmup_epochs": 10,
    "device": "cuda:0",

    # Hi-C loss weighting and automatic balancing
    "hic_weight": 1e-3,
    "auto_balance_hic_weight": True,
    "auto_balance_start_epoch": 5,
    "auto_balance_window": 5,

    # Anti-collapse regularization
    "anti_collapse_weight": 3.0,
    "anti_collapse_target_r": 0.75,
    "anti_collapse_start_epoch": 1,

    # Output and feature attribution
    "save_artifacts": False,
    "explain": False,
    "explain_top_k": 50,
    "explain_rep": "mu",
}


def demo(
    # Dataset and output paths
    hic_path,
    rna_path,
    output_dir,
    hic_resolution,
    epochs,
    save_root,

    # Model architecture and input features
    mode,
    n_genes,
    hic_hid_dim,
    rna_hid_dim,
    rna_fc_layers,
    latent_dim,
    z_dim,

    # Training
    batch_size,
    lr,
    max_beta,
    vmf_sampling_warmup_epochs,
    device,

    # Hi-C loss weighting and automatic balancing
    hic_weight,
    auto_balance_hic_weight,
    auto_balance_start_epoch,
    auto_balance_window,

    # Anti-collapse regularization
    anti_collapse_weight,
    anti_collapse_target_r,
    anti_collapse_start_epoch,

    # Output and feature attribution
    save_artifacts,
    explain,
    explain_top_k,
    explain_rep,
):
    os.makedirs(save_root, exist_ok=True)
    set_seed()
    torch.manual_seed(0)

    dataloader, dataset = create_multiomics_dataloader(
        hic_path,
        rna_path,
        batch_size,
        n_genes,
        shuffle=True,
        mode=mode,
    )
    print("Gene number: ", dataset.gene_num)
    fingerprint = {
        "seed": 0,
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
    print("Init")
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
        target_hic_ratio=hic_weight,
        auto_balance_after_first_epoch=auto_balance_hic_weight,
        auto_balance_start_epoch=auto_balance_start_epoch,
        auto_balance_window=auto_balance_window,
        vmf_sampling_warmup_epochs=vmf_sampling_warmup_epochs,
        anti_collapse_start_epoch=anti_collapse_start_epoch,
        mode=mode,
    )

    dataloader_eval = None
    if save_artifacts or explain:
        dataloader_eval = create_dataloader_from_dataset(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            mode=mode,
        )

    if save_artifacts:
        from evaluate import export_model_outputs
        model_config = {
            "hic_path": hic_path,
            "rna_path": rna_path,
            "output_dir": output_dir,
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
            "anti_collapse_weight": anti_collapse_weight,
            "anti_collapse_target_r": anti_collapse_target_r,
            "anti_collapse_start_epoch": anti_collapse_start_epoch,
            "vmf_sampling_warmup_epochs": vmf_sampling_warmup_epochs,
            "n_genes": n_genes,
            "max_beta": max_beta,
            "seed": 0,
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
        manifest["bundle_files"] = bundle_files
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Updated artifact manifest: {manifest_path}")

    if explain:
        from evaluate import explain_latent_features
        explain_latent_features(
            model=trained_model,
            dataloader_eval=dataloader_eval,
            dataset=dataset,
            device=device,
            save_root=save_root,
            rep_name=explain_rep,
            top_k=explain_top_k,
        )

    # Written only after training and all requested exports have succeeded.
    with open(os.path.join(save_root, "run_complete.json"), "w") as f:
        json.dump({"epochs": epochs, "seed": 0}, f, indent=2)
    return trained_model


def build_save_paths(current_params):
    batch_size = current_params["batch_size"]
    epochs = current_params["epochs"]
    lr = current_params["lr"]
    hic_resolution = current_params["hic_resolution"]
    z_dim = current_params["z_dim"]
    hic_weight = current_params["hic_weight"]
    auto_balance_hic_weight = current_params["auto_balance_hic_weight"]
    auto_balance_start_epoch = current_params["auto_balance_start_epoch"]
    auto_balance_window = current_params["auto_balance_window"]
    anti_collapse_weight = current_params["anti_collapse_weight"]
    anti_collapse_target_r = current_params["anti_collapse_target_r"]
    anti_collapse_start_epoch = current_params["anti_collapse_start_epoch"]
    vmf_sampling_warmup_epochs = current_params["vmf_sampling_warmup_epochs"]
    hic_hid_dim = current_params["hic_hid_dim"]
    rna_fc_layers = current_params["rna_fc_layers"]
    n_genes = current_params["n_genes"]
    max_beta = current_params["max_beta"]
    mode = current_params["mode"]
    output_dir = current_params.get("output_dir", "./hires_brain")

    base = (
        f"{output_dir.rstrip('/')}/"
        f"seed0_batch{batch_size}_epochs{epochs}_lr{lr}_"
        f"hicreso{hic_resolution}_zdim{z_dim}_hicratio{hic_weight}_hichiddim{hic_hid_dim}_"
        f"rnafclayers{rna_fc_layers}_gmmhvg{n_genes}_maxbeta{max_beta}_"
        f"hicbal{int(auto_balance_hic_weight)}e{auto_balance_start_epoch}w{auto_balance_window}"
        "clip0.001-1000.0_"
        f"collapse{anti_collapse_weight}r{anti_collapse_target_r}s{anti_collapse_start_epoch}_"
        f"vmfwarm{vmf_sampling_warmup_epochs}_"
        f"{mode}-poisson_vmf_spherical_stable"
    )
    return base


def run_experiment(overrides=None, skip_existing=False):
    params = {**DEFAULT_PARAMS, **(overrides or {})}
    save_root = build_save_paths(params)
    params["save_root"] = save_root

    completion_path = os.path.join(save_root, "run_complete.json")
    if skip_existing and os.path.exists(completion_path):
        print("Skip: run_complete.json already exists:", completion_path)
        return None

    print("Parameters:", params)
    model = demo(**params)
    print("=== Done ===")
    return model
