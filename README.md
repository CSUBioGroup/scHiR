# scHiR

scHiR learns joint representations of paired single-cell RNA and Hi-C data using a variational autoencoder with a von Mises–Fisher (vMF) latent distribution. It supports joint embedding, reconstruction of both modalities, and gradient-based latent feature attribution.

The included configuration uses the HiRES brain dataset. Training requires paired RNA and Hi-C inputs; no cell-type labels or metadata file is required.

## Requirements

scHiR was developed and tested using the following software environment:

- Python 3.13.12
- PyTorch 2.6.0 (CUDA 12.4)
- NumPy 2.2.4
- Pandas 2.2.3
- Scanpy 1.11.0
- SciPy 1.15.2
- Scikit-learn 1.5.2
- tqdm 4.67.1

### Installation

Create and activate a Conda environment:

```bash
conda create -n scHiR python=3.13
conda activate scHiR
```

Install PyTorch with CUDA 12.4 support:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies from the repository root:

```bash
pip install -r requirements.txt
```

**Note:** The listed versions correspond to the environment used for the experiments. GPU execution requires a compatible NVIDIA driver and CUDA-enabled PyTorch installation. Users may install a different version depending on their hardware, operating system, and CUDA requirements. See the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/) for platform-specific commands.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── main.py          # Command-line entry point
│   ├── configs.py       # Dataset paths and number of training epochs
│   ├── experiment.py    # Grouped default parameters and experiment execution
│   ├── data.py          # Input loading and preprocessing
│   ├── models.py        # Encoders, decoders, and joint VAE
│   ├── losses.py        # RNA and Hi-C reconstruction losses
│   ├── vmf.py           # vMF sampling and KL divergence
│   ├── train.py         # Training loop and automatic loss balancing
│   ├── evaluate.py      # Model export and latent feature attribution
│   └── utils.py         # Fixed random-state setup and reparameterization
└── data/
    ├── Hires_brain_cnn_1mb_sep_noimpute.pkl
    └── GSE223917_HiRES_brain.rna.umicount.tsv
```

## Input data

Place the two prepared input files in `data/` at the repository root. Default paths are resolved relative to the repository, independently of the working directory. The code consumes prepared chromosome contact matrices rather than raw Hi-C reads.

### Hi-C input

The `.pkl` file must contain a cell-ordered collection. Each cell contains a chromosome-ordered collection of dense, two-dimensional NumPy contact matrices:

```python
hic_data[cell_index][chromosome_index]  # shape: (n_bins, n_bins)
```

Use the same chromosome order across cells and the same matrix size for each corresponding chromosome. For each matrix, the code crops both dimensions to multiples of four, converts to `float32`, applies `log1p`, clips at the 99th percentile, and divides by its maximum plus `1e-6`. This preprocessing is fixed in the code.

### RNA input

The RNA file must be a gene-by-cell count table, with gene names in the first column and cell names in the header. Tab- and comma-delimited files are supported through automatic delimiter detection.

**The RNA columns must have exactly the same cell order as the Hi-C collection.** The loader checks cell counts but does not match or reorder cells across modalities.

For feature selection, the code removes genes expressed in no cells, normalizes a copy of the RNA data to 10,000 counts per cell, applies `log1p`, and selects highly variable genes using Scanpy's `seurat` method. The encoder receives the original counts restricted to the selected genes. The default is 2,000 genes, and the loader requires exactly the requested number to be selected.

## Usage

Run the following commands from the repository root.

### Train and save results

```bash
python code/main.py --dataset hires_brain --save-artifacts
```

The default configuration trains for 100 epochs on `cuda:0`. Results are written to a parameter-specific subdirectory under `hires_brain/`.

To select another GPU:

```bash
python code/main.py --save-artifacts --device cuda:1
```

To run on CPU:

```bash
python code/main.py --save-artifacts --device cpu
```

Running `python code/main.py` without `--save-artifacts` trains the model and saves training logs and fingerprints, but does not save model checkpoints, embeddings, or reconstructed outputs.

### Latent feature attribution

```bash
python code/main.py --save-artifacts --explain --explain-top-k 50
```

Attribution uses gradients multiplied by inputs and aggregates absolute contributions across cells. By default, it explains the joint `mu` representation and saves the top 50 RNA genes per latent dimension and top 50 Hi-C contacts per chromosome per latent dimension, together with modality contribution summaries.

### Custom input and output paths

```bash
python code/main.py \
    --hic-path data/custom_hic.pkl \
    --rna-path data/custom_rna.tsv \
    --output-dir results/custom \
    --save-artifacts
```

Explicit relative path overrides are resolved from the current working directory. Custom inputs must follow the same data format and cell-order requirements.

### Skip completed runs

```bash
python code/main.py --save-artifacts --skip-existing
```

This skips a run when its output directory contains `run_complete.json`. It does not resume interrupted training or check whether a previous run saved all artifacts requested by the current command.

To list command-line options:

```bash
python code/main.py --help
```

## Configuration

Dataset paths, the output directory, the resolution label, and the number of epochs are defined in `code/configs.py`. Model and training defaults are grouped in `DEFAULT_PARAMS` in `code/experiment.py`:

| Group | Parameter | Default |
| --- | --- | --- |
| Model and features | `mode` | `multi` |
| Model and features | `n_genes` | `2000` |
| Model and features | `hic_hid_dim` | `200` |
| Model and features | `rna_hid_dim` | `128` |
| Model and features | `rna_fc_layers` | `3` |
| Model and features | `latent_dim` | `128` |
| Model and features | `z_dim` | `32` |
| Training | `batch_size` | `32` |
| Training | `lr` | `5e-4` |
| Training | `max_beta` | `0.05` |
| Training | `vmf_sampling_warmup_epochs` | `10` |
| Training | `device` | `cuda:0` |
| Hi-C weighting | `hic_weight` | `1e-3` |
| Hi-C weighting | `auto_balance_hic_weight` | `True` |
| Hi-C weighting | `auto_balance_start_epoch` | `5` |
| Hi-C weighting | `auto_balance_window` | `5` |
| Anti-collapse | `anti_collapse_weight` | `3.0` |
| Anti-collapse | `anti_collapse_target_r` | `0.75` |
| Anti-collapse | `anti_collapse_start_epoch` | `1` |
| Output and attribution | `save_artifacts` | `False` |
| Output and attribution | `explain` | `False` |
| Output and attribution | `explain_top_k` | `50` |
| Output and attribution | `explain_rep` | `mu` |

`latent_dim` is the intermediate encoder dimension; `z_dim` is the joint embedding dimension. The initial learning rate is adjusted by the existing plateau scheduler during the second half of training. vMF posterior sampling begins after the first 10 epochs; the KL coefficient follows a separate schedule based on the total number of epochs.

Automatic Hi-C balancing updates the weight once, normally after epoch 5. It uses the median of the latest five candidate weights, each computed as `hic_weight × RNA loss / raw Hi-C loss`. The weight is clipped between `hic_weight × 0.001` and `hic_weight × 1000.0`; these bounds are fixed in the training code.

`hic_resolution` and the `--hic-resolution` option provide a label for output naming and configuration records. They do not resample or change the resolution of the input contact matrices.

## Saved outputs

With `--save-artifacts`, the run directory contains the following outputs, where `<epoch>` is the final training epoch:

| Output | Contents |
| --- | --- |
| `model_full_epoch<epoch>.pt` | Serialized model |
| `model_checkpoint_epoch<epoch>.pt` | Model state dictionary and configuration |
| `model_config_epoch<epoch>.json` | Model and training settings, dimensions, and data fingerprints |
| `embedding_<rep>_epoch<epoch>.npy` / `.csv` | Latent representations in input cell order |
| `input_encoder_scrna_epoch<epoch>.npy` | Selected RNA counts supplied to the encoder |
| `input_encoder_scrna_genes_epoch<epoch>.txt` | Selected gene names in encoder input order |
| `input_encoder_schic_epoch<epoch>.npz` | Preprocessed Hi-C encoder inputs |
| `input_encoder_schic_chr<idx>_epoch<epoch>.npy` | Hi-C encoder inputs for each chromosome |
| `output_scrna_epoch<epoch>.npy` | Decoded RNA rates |
| `output_schic_epoch<epoch>.npz` | Hi-C decoder outputs |
| `output_schic_chr<idx>_epoch<epoch>.npy` | Hi-C decoder outputs for each chromosome |
| `artifact_manifest.json` / `repro_bundle_files.txt` | Artifact descriptions and file listing |

In joint mode, exported representations include `mu`, `pre_spherical`, `concentration`, `mu_scaled`, `z`, `rna_mu`, and `hic_mu`. The primary joint embedding is `mu`, the unit-normalized vMF direction. With the default configuration, it is saved as `embedding_mu_epoch100.npy` and `embedding_mu_epoch100.csv`, with shape `(n_cells, 32)`.

Hi-C reconstruction files contain raw decoder outputs. The training loss applies `softplus` plus `1e-8` to obtain positive Poisson rates. RNA reconstruction files already contain decoded rates.

Training also writes `train_history_seed0.csv`, `run_fingerprint.json`, and `first_batch_fingerprint_seed0.json`. A successful run writes `run_complete.json` after all requested operations finish. Checkpoints contain the trained model, but not optimizer or scheduler state for resuming training.

With `--explain`, additional files are saved:

- `latent_feature_mu_rna_top.csv`
- `latent_feature_mu_hic_top.csv`
- `latent_feature_mu_modality_summary.csv`

## Reproducibility

Random states are fixed in the code with a base seed of 0. DataLoader workers use `0 + worker_id`. Deterministic PyTorch settings are enabled, and TF32 is disabled.

Run fingerprints record the software environment and data checksums. SHA256 fields track cell names and order, selected gene names and order, and the RNA encoder input matrix. These are diagnostic records and do not participate in training.

For reproducibility, retain the same input data, cell and gene ordering, configuration, software environment, and hardware. A fixed seed alone does not guarantee identical results across different platforms or library versions.
