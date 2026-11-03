# scHiR

**Single-cell RNA and Hi-C representation learning with a variational autoencoder.**

scHiR is a research-oriented PyTorch implementation for learning low-dimensional cell representations from single-cell RNA sequencing (scRNA-seq) and single-cell Hi-C (scHi-C) data. The codebase includes modality-specific encoders and decoders, a von Mises–Fisher (vMF) latent representation, training and evaluation routines, and optional latent-feature attribution.

> **Status:** Research code. Dataset paths in `configs.py` refer to the original computing environment and must be replaced before running the experiments. Preprocessed data and a pinned dependency manifest are not currently included in this repository.

## Repository structure

| File | Description |
| --- | --- |
| `main.py` | Command-line entry point for running dataset experiments. |
| `configs.py` | Dataset presets, input locations, labels, cluster counts, and output settings. |
| `data.py` | Data loading and preprocessing for RNA and Hi-C modalities. |
| `models.py` | Model architecture, modality encoders/decoders, and VAE components. |
| `vmf.py` | vMF distribution/sampling utilities. |
| `losses.py` | Reconstruction and related loss functions. |
| `train.py` | Model training and artifact handling. |
| `evaluate.py` | Evaluation utilities for learned representations. |
| `experiment.py` | Experiment configuration, parameter combinations, and execution. |
| `utils.py` | Shared helper functions. |

## Installation

Clone the repository:

```bash
git clone https://github.com/zhaoyichao777/scHiR.git
cd scHiR
```

Create a Python environment and install the scientific Python packages used by the project. For example:

```bash
conda create -n schir python=3.10 -y
conda activate schir

# Install a PyTorch build compatible with your system/CUDA setup.
# See https://pytorch.org/get-started/locally/
pip install numpy pandas scipy scikit-learn scanpy anndata matplotlib seaborn tqdm
```

**Important:** The command above is a starting point, not a validated complete environment specification. Install PyTorch separately and check the imports in the source files for any additional packages. Package versions have not been pinned or tested as a reproducible release.

## Data preparation

The experiment runner expects paths to the following inputs:

1. **Cell metadata:** a CSV containing cell annotations and the label column(s) specified by the dataset configuration.
2. **scRNA-seq data:** a gene-by-cell/cell-by-gene count file in the format expected by `data.py`.
3. **scHi-C data:** preprocessed per-cell contact matrices stored in the serialized format expected by `data.py`.

Before execution, edit the paths in `configs.py` or override them on the command line. Input files are **not** bundled with this repository. Ensure that RNA, Hi-C, and metadata refer to matching cells and that preprocessing conforms to the loaders in `data.py`.

## Quick start

Inspect the command-line options:

```bash
python main.py --help
```

Run experiments using one of the provided dataset presets after updating its paths:

```bash
python main.py --dataset limca_hg38
```

Alternatively, override the input and output paths without changing the preset:

```bash
python main.py \
  --dataset limca_hg38 \
  --metadata-path /path/to/metadata.csv \
  --rna-path /path/to/rna_counts.csv \
  --hic-path /path/to/hic_matrices.pkl \
  --output-dir ./results/limca_hg38 \
  --device cuda:0
```

For an evaluation-free run with artifact saving enabled:

```bash
python main.py \
  --dataset limca_hg38 \
  --no-eval \
  --save-artifacts
```

For latent-feature attribution, add `--explain` and optionally adjust `--explain-top-k`:

```bash
python main.py --dataset limca_hg38 --explain --explain-top-k 50
```

These examples assume that the required files exist and are correctly formatted; they have not been verified end-to-end in a clean environment.

## Experiment configuration

Dataset-specific paths and settings live in `configs.py`. The hyperparameter search and model defaults live in `experiment.py`.

The default parameter grid currently includes:

| Parameter | Default values |
| --- | --- |
| Learning rate | `5e-4` |
| Batch size | `32` |
| Latent size (`z_dim`) | `32` |
| Selected genes (`n_genes`) | `2000` |
| Modes | `rna_only`, `hic_only` |

Additional fixed settings, including the number of epochs, vMF warmup, and automatic Hi-C loss weighting, are defined in the code. To run `multi` mode, update the `mode` entries in `DEFAULT_PARAM_GRID` in `experiment.py`.

## Outputs

Experiments write results under the configured output directory, with subdirectories constructed from the experiment settings. Depending on the run options, outputs can include checkpoints, learned embeddings, decoded modality outputs, and explanation artifacts.

Artifact saving is disabled by default in the current experiment configuration. Use `--save-artifacts` to enable it.

## Citation

If you use this code in academic work, please cite the associated publication when its bibliographic information becomes available. 
