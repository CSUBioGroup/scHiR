from pathlib import Path


# Resolve repository-relative paths independently of the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIGS = {
    "hires_brain": {
        "hic_path": "data/Hires_brain_cnn_1mb_sep_noimpute.pkl",
        "rna_path": "data/GSE223917_HiRES_brain.rna.umicount.tsv",
        "output_dir": "./hires_brain",
        "epochs": 100,
    },
}


def get_dataset_config(name):
    if name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    config = DATASET_CONFIGS[name].copy()
    for key in ("hic_path", "rna_path", "output_dir"):
        config[key] = str(PROJECT_ROOT / config[key])
    return config
