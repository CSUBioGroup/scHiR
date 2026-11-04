import argparse
import sys

from configs import DATASET_CONFIGS, get_dataset_config
from experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-omics VAE experiments.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="hires_brain",
        help="Dataset preset from configs.py.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=None,
        help="Run latent feature attribution after training.",
    )
    parser.add_argument(
        "--explain-top-k",
        type=int,
        default=None,
        help="Top-k RNA genes / Hi-C contacts to save per latent dimension.",
    )
    parser.add_argument(
        "--save-artifacts",
        dest="save_artifacts",
        action="store_true",
        default=None,
        help="Save all representations, trained model, processed inputs, and decoded outputs.",
    )


    parser.add_argument("--hic-path", default=None)
    parser.add_argument("--rna-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Python executable:", sys.executable)

    dataset_config = get_dataset_config(args.dataset)
    if args.hic_path is not None:
        dataset_config["hic_path"] = args.hic_path
    if args.rna_path is not None:
        dataset_config["rna_path"] = args.rna_path
    if args.output_dir is not None:
        dataset_config["output_dir"] = args.output_dir

    overrides = {
        "hic_path": dataset_config["hic_path"],
        "rna_path": dataset_config["rna_path"],
        "output_dir": dataset_config["output_dir"],
        "epochs": dataset_config["epochs"],
    }

    if args.explain is not None:
        overrides["explain"] = args.explain
    if args.explain_top_k is not None:
        overrides["explain_top_k"] = args.explain_top_k
    if args.device is not None:
        overrides["device"] = args.device
    if args.save_artifacts is not None:
        overrides["save_artifacts"] = args.save_artifacts

    print("Dataset:", args.dataset)
    print("Output dir:", dataset_config["output_dir"])

    run_experiment(
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
