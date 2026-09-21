import argparse
import sys

import pandas as pd

from configs import DATASET_CONFIGS, get_dataset_config
from experiment import hyperparameter_tuning


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-omics VAE experiments.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        help="Dataset preset from configs.py.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable final evaluation.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run latent feature attribution after training.",
    )
    parser.add_argument(
        "--explain-top-k",
        type=int,
        default=50,
        help="Top-k RNA genes / Hi-C contacts to save per latent dimension.",
    )
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument(
        "--save-artifacts",
        dest="save_artifacts",
        action="store_true",
        default=None,
        help="Save trained model, embeddings, and decoded scRNA/scHiC outputs.",
    )
    artifact_group.add_argument(
        "--no-save-artifacts",
        dest="save_artifacts",
        action="store_false",
        help="Do not save trained model, embeddings, or decoded scRNA/scHiC outputs.",
    )

    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--label-column", action="append", dest="label_columns")
    parser.add_argument("--hic-path", default=None)
    parser.add_argument("--rna-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hic-resolution", default=None)
    parser.add_argument("--n-cluster", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing result CSV.")
    return parser.parse_args()


def parse_n_cluster_overrides(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--n-cluster must look like 'Cell type=4'")
        label_name, n_cluster = value.split("=", 1)
        overrides[label_name.strip()] = int(n_cluster)
    return overrides


def main():
    args = parse_args()
    print("Python executable:", sys.executable)

    dataset_config = get_dataset_config(args.dataset)
    if args.metadata_path is not None:
        dataset_config["metadata_path"] = args.metadata_path
    if args.label_columns is not None:
        dataset_config["label_columns"] = args.label_columns
    if args.hic_path is not None:
        dataset_config["hic_path"] = args.hic_path
    if args.rna_path is not None:
        dataset_config["rna_path"] = args.rna_path
    if args.output_dir is not None:
        dataset_config["output_dir"] = args.output_dir
    if args.hic_resolution is not None:
        dataset_config["hic_resolution"] = args.hic_resolution

    n_cluster = dict(dataset_config["n_cluster"])
    n_cluster.update(parse_n_cluster_overrides(args.n_cluster))
    dataset_config["n_cluster"] = n_cluster

    fixed_overrides = {
        "hic_path": dataset_config["hic_path"],
        "rna_path": dataset_config["rna_path"],
        "n_cluster": dataset_config["n_cluster"],
        "output_dir": dataset_config["output_dir"],
        "hic_resolution": dataset_config["hic_resolution"],
        "epochs": dataset_config["epochs"],
        "cal": not args.no_eval,
        "explain": args.explain,
        "explain_top_k": args.explain_top_k,
    }

    if args.device is not None:
        fixed_overrides["device"] = args.device
    if args.save_artifacts is not None:
        fixed_overrides["save_artifacts"] = args.save_artifacts

    metadata = pd.read_csv(dataset_config["metadata_path"])
    labels = metadata[dataset_config["label_columns"]]

    print("Dataset:", args.dataset)
    print("Labels:", dataset_config["label_columns"])
    print("Output dir:", dataset_config["output_dir"])

    hyperparameter_tuning(
        labels,
        fixed_overrides=fixed_overrides,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
