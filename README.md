# PLAID (Protein Latent Induced Diffusion)

## Contents

- [Contents](#contents)
- [Demo](#demo)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Environment Setup](#environment-setup)
  - [Model Weights](#model-weights)
  - [Loading Pretrained Models](#loading-pretrained-models)
- [Usage](#usage)
  - [Example Quick Start](#example-quick-start)
  - [Full Pipeline](#full-pipeline)
  - [Design-Only Inference](#design-only-inference)
  - [Evaluation](#evaluation)
- [Training](#training)
- [License](#license)

## Demo

A hosted demo of the model will be available soon.


## Installation

### Clone the Repository

```bash
git clone https://github.com/amyxlu/plaid.git
cd plaid
```


### Environment Setup
Create the environment and install dependencies:

```bash
conda env create --file environment.yaml  # Create environment
pip install --no-deps git+https://github.com/amyxlu/openfold.git  # Install OpenFold
pip install -e .  # Install PLAID
```

Note: The OpenFold implementation of the ESMFold module includes custom CUDA kernels for the attention mechanism. This repository uses a fork of OpenFold with C++17 compatibility for CUDA kernels to support `torch >= 2.0`.


### Model Weights
* Latent Autoencoder (CHEAP): full codebase is available [here](https://github.com/amyxlu/cheap-proteins). We use the `CHEAP_pfam_shorten_2_dim_32()` model.
* Diffusion Weights (PLAID): Hosted on [HuggingFace](https://huggingface.co/amyxlu/plaid/tree/main). There is both a 2B and a 100M model.

By default, PLAID weights are cached in `~/.cache/plaid` and CHEAP latent autoencoder weights in `~/.cache/cheap`. Customize the cache path using:

```bash
echo "export CHEAP_CACHE=/path/to/cache" >> ~/.bashrc  # see CHEAP README for more details
echo "export PLAID_CACHE=/path/to/cache" >> ~/.bashrc
```

### Loading Pretrained Models

```python
from plaid.pretrained import PLAID_2B, PLAID_100M
denoiser, cfg = PLAID_2B()
```

This loads the PLAID DiT denoiser, and the hyperparameters used to initialize the diffusion object defined in `src/plaid/diffusion/cfg.py`.
The denoiser and diffusion configuration is loaded separately, since in theory, the denoiser can be used with any other diffusion setup, such as [EDM](https://github.com/lucidrains/edm-pytorch).
Using the sampling steps below will initialize the discrete diffusion process used in our paper.


## Usage

### Example Quick Start

```bash
python pipeline/run_pipeline.py experiment=unconditional_no_analysis
```

This experiment is specified in `configs/inference/experiment/unconditional_no_analysis.yaml`, which overrides settings in `configs/inference/full.yaml`.As the YAML name suggests, it runs unconditional sampling (Steps 1 and 2 in the [Design-Only Inference](#design-only-inference) section) without analysis (Step 3 in the [Evaluation](#evaluation) section).

**Most sampling parameters (e.g. GO term, organism, length) are specified in `configs/inference/sample/ddim_unconditional.yaml`. Update this config group for your needs. See Step 1 in the [Design-Only Inference](#design-only-inference) section for more details.**

### Full Pipeline
The entire `pipeline/run_pipeline.py` script will run the full pipeline, including sampling, decoding, consistency, and analysis (Steps 1-3 in the [Design-Only Inference](#design-only-inference) and [Evaluation](#evaluation) sections). You can turn off Steps 2 and 3, as documented in `configs/inference/full.yaml`. You can also run each of these steps as individual scripts, if you need to resume from a pipeline step after an error.

### Design-Only Inference
PLAID generation consists of:
1. Sampling latent embeddings.
2. Decoding these embeddings into sequences and structures.

#### Step 1: Sampling Latent Embeddings
1. Run latent sampling using Hydra-configured scripts in configs/pipeline/sample/. Example commands:

```bash
# Conditional sampling with inferred length
python pipeline/run_sample.py ++length=null ++function_idx=166 ++organism_idx=1326

# Conditional sampling with fixed length
python pipeline/run_sample.py ++length=200 ++function_idx=166 ++organism_idx=1326

# Unconditional sampling with specified output directory
python pipeline/run_sample.py ++length=200 ++function_idx=2219 ++organism_idx=3617 ++output_root_dir=/data/lux70/plaid/samples/unconditional
```

>[!IMPORTANT]
>The specified length is half the actual protein length and must be divisible by 4. For example, to generate a 200-residue protein, set length=100.

>[!TIP]
>To find the mapping between your desired GO term and function index, see `src/plaid/constants.py`. A list of organism indices can be found in `assets/organisms`.

>[!TIP]
>PLAID also supports the DPM++ sampler, which achieves comparable performance with fewer sampling steps. See `configs/inference/sample/dpm2m_sde.yaml` for more details.

#### Step 2: Decode the Latent Embedding
* 2a. Uncompress latent arrays using the CHEAP autoencoder.
* 2b. Use the CHEAP sequence decoder for sequences.
* 2c. Use the ESMFold structure encoder for structures.


## Evaluation
Reproduce results or perform advanced analyses using the evaluation pipeline. Steps:

3. Generate inverse and phantom sequences/structures:

```bash
python pipeline/run_consistency.py ++samples_dir=/path/to/samples
```

4. Analyze metrics (ccRMSD, novelty, diversity, etc.):

```bash
python pipeline/run_analysis.py /path/to/samples
```


## Training
Train PLAID models using PyTorch Lightning with distributed data parallel (DDP). Example launch command for training on 8 A100 GPUs:

```bash
python train_compositional.py  # see config/experiments
```

Key features:

* Min-SNR loss scaling
* Classifier-free guidance (GO terms and organisms)
* Self-conditioning
* EMA weight decay

Note: If using torch.compile, ensure precision is set to float32 due to compatibility issues with the xFormers library.

Embeddings are pre-computed and cached as `.tar` files for compatibility with [WebDataset](https://github.com/webdataset/webdataset) dataloaders. Pfam embedding `.tar` files used for training and validation data will be uploaded soon.

## License

PLAID is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
