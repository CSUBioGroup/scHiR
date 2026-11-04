import os
import json
import hashlib
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def move_batch(data, model, device):
    rna = None if model.mode == "hic_only" else data["rna"].to(device, non_blocking=True)
    hic = None if model.mode == "rna_only" else [
        x.to(device, non_blocking=True) for x in data["hic"]
    ]
    return rna, hic


def _tensor_sha256(tensor):
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def train_multiomics(
    model,
    dataloader,
    epochs,
    lr,
    save_root,
    max_beta,
    target_hic_ratio,
    auto_balance_after_first_epoch=True,
    auto_balance_start_epoch=5,
    auto_balance_window=5,
    vmf_sampling_warmup_epochs=10,
    anti_collapse_start_epoch=1,
    device="cuda",
    mode="multi"
):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=3,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1e-5,
    )

    history_rows = []
    balance_candidates = []
    hic_weight_balanced = False
    base_anti_collapse_weight = getattr(model, "anti_collapse_weight", 0.0)
    first_batch_fingerprint_written = False

    if save_root:
        os.makedirs(save_root, exist_ok=True)
        history_csv = os.path.join(
            save_root,
            "train_history.csv"
        )
    else:
        history_csv = None

    print("=== start training ===")
    print("cuda available:", torch.cuda.is_available())

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        current_epoch = epoch + 1
        beta = max_beta * min(1.0, epoch / max(epochs, 1))
        sample_vmf_posterior = current_epoch > vmf_sampling_warmup_epochs
        anti_collapse_active = (
            model.mode == "multi"
            and current_epoch >= anti_collapse_start_epoch
        )
        model.anti_collapse_weight = (
            base_anti_collapse_weight if anti_collapse_active else 0.0
        )
        stats = {
            "total_loss": 0.0,
            "rna_loss": 0.0,
            "raw_hic_loss": 0.0,
            "weighted_hic_loss": 0.0,
            "kl_loss": 0.0,
            "anti_collapse_loss": 0.0,
            "anti_collapse_r": 0.0,
        }

        model.train()

        for num_batches, data in enumerate(dataloader, start=1):
            rna, hic = move_batch(data, model, device)

            optimizer.zero_grad(set_to_none=True)

            (
                loss,
                rna_loss,
                raw_hic_loss,
                weighted_hic_loss,
                kl_loss,
                anti_collapse_loss,
                anti_collapse_r,
            ) = model(
                rna,
                hic,
                beta,
                sample_vmf_posterior=sample_vmf_posterior,
            )

            if (
                save_root
                and not first_batch_fingerprint_written
                and current_epoch == 1
                and num_batches == 1
            ):
                first_param = next(model.parameters()).detach()
                first_batch_fingerprint = {
                    "seed": 0,
                    "epoch": current_epoch,
                    "batch": num_batches,
                    "beta": beta,
                    "sample_vmf_posterior": sample_vmf_posterior,
                    "anti_collapse_active": anti_collapse_active,
                    "first_param_sum": float(first_param.cpu().double().sum().item()),
                    "first_param_sha256": _tensor_sha256(first_param),
                    "rna_sum": None if rna is None else float(rna.detach().cpu().double().sum().item()),
                    "rna_sha256": None if rna is None else _tensor_sha256(rna.detach()),
                    "hic_sums": None if hic is None else [
                        float(x.detach().cpu().double().sum().item()) for x in hic
                    ],
                    "hic_sha256": None if hic is None else [
                        _tensor_sha256(x.detach()) for x in hic
                    ],
                    "loss": float(loss.detach().cpu().item()),
                    "rna_loss": float(rna_loss.detach().cpu().item()),
                    "raw_hic_loss": float(raw_hic_loss.detach().cpu().item()),
                    "weighted_hic_loss": float(weighted_hic_loss.detach().cpu().item()),
                    "kl_loss": float(kl_loss.detach().cpu().item()),
                    "anti_collapse_loss": float(anti_collapse_loss.detach().cpu().item()),
                    "anti_collapse_r": float(anti_collapse_r.detach().cpu().item()),
                }
                fingerprint_path = os.path.join(
                    save_root,
                    "first_batch_fingerprint.json",
                )
                with open(fingerprint_path, "w") as f:
                    json.dump(
                        first_batch_fingerprint,
                        f,
                        indent=2,
                        sort_keys=True,
                    )
                print("Saved first batch fingerprint to:", fingerprint_path)
                first_batch_fingerprint_written = True

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )
            optimizer.step()        
           

            stats["total_loss"] += loss.item()
            stats["rna_loss"] += rna_loss.item()
            stats["raw_hic_loss"] += raw_hic_loss.item()
            stats["weighted_hic_loss"] += weighted_hic_loss.item()
            stats["kl_loss"] += kl_loss.item()
            stats["anti_collapse_loss"] += anti_collapse_loss.item()
            stats["anti_collapse_r"] += anti_collapse_r.item()

        avg = {
            k: v / max(num_batches, 1)
            for k, v in stats.items()
        }
        balance_candidate = None
        if (
            model.mode not in {"rna_only", "hic_only"}
            and avg["raw_hic_loss"] > 0
        ):
            balance_candidate = float(
                target_hic_ratio
                * avg["rna_loss"]
                / avg["raw_hic_loss"]
            )
            balance_candidates.append(balance_candidate)

        # Auto-balance hic_weight after a short warmup. Using a median over
        # several epochs avoids letting one seed-specific first epoch set the
        # whole multi-omic weighting.
        if (
            model.mode not in {"rna_only", "hic_only"}
            and auto_balance_after_first_epoch
            and not hic_weight_balanced
            and current_epoch >= auto_balance_start_epoch
            and len(balance_candidates) >= auto_balance_window
            and avg["raw_hic_loss"] > 0
        ):
            recent_candidates = balance_candidates[-auto_balance_window:]
            sorted_candidates = sorted(recent_candidates)
            mid = len(sorted_candidates) // 2
            if len(sorted_candidates) % 2 == 0:
                new_hic_weight = 0.5 * (
                    sorted_candidates[mid - 1] + sorted_candidates[mid]
                )
            else:
                new_hic_weight = sorted_candidates[mid]

            unclipped_hic_weight = float(new_hic_weight)
            min_hic_weight = target_hic_ratio * 0.001
            max_hic_weight = target_hic_ratio * 1000.0
            new_hic_weight = min(
                max(new_hic_weight, min_hic_weight),
                max_hic_weight,
            )

            model.hic_weight = float(new_hic_weight)
            hic_weight_balanced = True

            print(
                f"[Auto hic_weight] epoch={current_epoch} | "
                f"median_candidate={unclipped_hic_weight:.8f} | "
                f"new_hic_weight={model.hic_weight:.8f} | "
                f"clip_range=[{min_hic_weight:.8f}, {max_hic_weight:.8f}] | "
                f"window={auto_balance_window}"
            )

        row = {
            "epoch": current_epoch,
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            "hic_weight": getattr(model, "hic_weight", 0.0),
            "hic_weight_balance_candidate": balance_candidate,
            "hic_weight_balanced": hic_weight_balanced,
            "sample_vmf_posterior": sample_vmf_posterior,
            "anti_collapse_start_epoch": anti_collapse_start_epoch,
            "anti_collapse_active": anti_collapse_active,
            "anti_collapse_weight_effective": getattr(
                model,
                "anti_collapse_weight",
                0.0,
            ),
            **avg,
        }

        history_rows.append(row)

        if history_csv:
            pd.DataFrame(history_rows).to_csv(
                history_csv,
                index=False
            )

        if current_epoch >= epochs/2:
            scheduler.step(
                avg["rna_loss"] + avg["weighted_hic_loss"]
            )
    model.anti_collapse_weight = base_anti_collapse_weight
    torch.cuda.empty_cache()
    return model
