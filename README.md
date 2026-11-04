# scHiR

scHiR learns joint representations of paired single-cell RNA and Hi-C data using a variational autoencoder with a von Mises–Fisher (vMF) latent distribution. It supports joint embedding, reconstruction of both modalities, and gradient-based latent feature attribution.

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

### scHi-C input

The `.pkl` file must contain a cell-ordered collection. Each cell contains a chromosome-ordered collection of dense, two-dimensional NumPy contact matrices:

```python
hic_data[cell_index][chromosome_index]  # shape: (n_bins, n_bins)
```

Use the same chromosome order across cells and the same matrix size for each corresponding chromosome. 

### scRNA-seq input

The scRNA-seq file must be a gene-by-cell count table, with gene names in the first column and cell names in the header. Tab- and comma-delimited files are supported through automatic delimiter detection.

**The RNA columns must have exactly the same cell order as the Hi-C collection.** The loader checks cell counts but does not match or reorder cells across modalities.

## Usage

Run the following commands from the repository root.

### Train and save embeddings

```bash
python code/main.py --dataset hires_brain
```

By default, scHiR saves the joint `mu` embedding in `.npy` and `.csv` formats, together with training logs and fingerprints. Model checkpoints, other representations, processed input arrays, and reconstructed outputs are saved only when `--save-artifacts` is specified:

```bash
python code/main.py --save-artifacts
```

The default configuration trains for 100 epochs on `cuda:0`. Results are written to a timestamped subdirectory such as `hires_brain/results_20261104_163025/`, using the local time of the machine running training. Runs started in the same second receive a numeric suffix to avoid overwriting existing results.

To select another GPU:

```bash
python code/main.py --save-artifacts --device cuda:1
```

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

To list command-line options:

```bash
python code/main.py --help
```

## Saved outputs

By default, the run directory contains `embedding_mu_epoch<epoch>.npy` and `embedding_mu_epoch<epoch>.csv`, as well as training logs, fingerprints, and the completion marker. Here, `<epoch>` is the final training epoch.

With `--save-artifacts`, it additionally contains the full set of representations, model files, processed inputs, and reconstructed outputs listed below:

| Output | Contents |
| --- | --- |
| `model_full_epoch<epoch>.pt` | Serialized model |
| `model_checkpoint_epoch<epoch>.pt` | Model state dictionary and configuration |
| `model_config_epoch<epoch>.json` | Model and training settings, dimensions, and data fingerprints |
| `embedding_<rep>_epoch<epoch>.npy` / `.csv` | Latent representations in input cell order |
| `input_encoder_scrna_epoch<epoch>.npz` | Selected RNA counts (`counts`) and gene names in encoder input order (`gene_names`) |
| `input_encoder_schic_epoch<epoch>.npz` | Preprocessed Hi-C encoder inputs |
| `input_encoder_schic_chr<idx>_epoch<epoch>.npy` | Hi-C encoder inputs for each chromosome |
| `output_scrna_epoch<epoch>.npy` | Decoded RNA rates |
| `output_schic_epoch<epoch>.npz` | Hi-C decoder outputs |
| `output_schic_chr<idx>_epoch<epoch>.npy` | Hi-C decoder outputs for each chromosome |
| `artifact_manifest.json` / `repro_bundle_files.txt` | Artifact descriptions and file listing |

With `--save-artifacts` in joint mode, exported representations include `mu`, `pre_spherical`, `concentration`, `mu_scaled`, `z`, `rna_mu`, and `hic_mu`. The primary joint embedding is `mu`, the unit-normalized vMF direction. With the default configuration, it is saved as `embedding_mu_epoch100.npy` and `embedding_mu_epoch100.csv`, with shape `(n_cells, 32)`.

Hi-C reconstruction files contain raw decoder outputs. The training loss applies `softplus` plus `1e-8` to obtain positive Poisson rates. RNA reconstruction files already contain decoded rates.

Training also writes `train_history.csv`, `run_fingerprint.json`, and `first_batch_fingerprint.json`. A successful run writes `run_complete.json` after all requested operations finish. Checkpoints contain the trained model, but not optimizer or scheduler state for resuming training.

With `--explain`, additional files are saved:

- `latent_feature_mu_rna_top.csv`
- `latent_feature_mu_hic_top.csv`
- `latent_feature_mu_modality_summary.csv`
