# Scoliosis Syn2Real

Code and input-data preparation resources for grouped synthetic-to-real Cobb-angle band classification.

This repository provides three training workflows: real-only cross-entropy training, synthetic pretraining followed by real-target fine-tuning, and naive joint synthetic/real training. They use a common DINOv2 ViT-S/14 backbone and matched target-stage exposure and optimizer-update budgets. Source exposure and total computation are **not** matched.

## Public release scope

Included: training and evaluation functions, configuration, label-free dataset identities and split/budget assignments, data preparation tools, and unit tests.

`engine/statistics.py` and `engine/sensitivity.py` also expose the original numerical routines for paired group/seed bootstrap, Holm adjustment and confidence-ranked risk-coverage calculations, without any observed predictions or results. Their low-level array interfaces require aligned case order, truth labels and containment groups across every workflow/seed. Fields named `empirical_coverage` in risk-coverage outputs mean the retained image fraction, not conformal interval coverage; tail-proportion bootstrap fields do not establish clinical validity.

Excluded: manuscripts, figures, tables, experimental results, predictions, training logs, trained checkpoints, credentials, and original third-party image/label collections. The `data/` directory is metadata, **not a new copy of either dataset**. See [data/README.md](data/README.md).

## Environment

Use Python 3.10+ on a CUDA server. Install a mutually compatible CUDA-enabled PyTorch/torchvision pair from [PyTorch](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_data.py --check-metadata
python run.py plan
```

The training configuration requires CUDA bfloat16 support. The local release checks do not launch GPU training. The portable adapter has not yet been validated by a new end-to-end GPU run; do not interpret passing unit tests as numerical reproduction of an experiment.

## Acquire inputs

1. Obtain the AASCE/SpineWeb `boostnet_labeldata` collection from an appropriate upstream source, preserving its `data/training` and `labels/training` layout.
2. Obtain [Spinal-AI2024](https://github.com/Ernestchenchen/Spinal-AI2024), including subsets 1-4 and `Cobb_spinal-AI2024-train_gt.txt`. The source reference used here is commit `693acc5aee5620e9b25dcc15accdaff42d1f930e`.
3. Obtain the official `dinov2_vits14_pretrain.pth` file through the [DINOv2 project](https://github.com/facebookresearch/dinov2). Its required SHA-256 is in `configs/training.json`. Third-party model weights are not included.

```bash
git clone --filter=blob:none --no-checkout https://github.com/Ernestchenchen/Spinal-AI2024.git /path/to/Spinal-AI2024
git -C /path/to/Spinal-AI2024 checkout 693acc5aee5620e9b25dcc15accdaff42d1f930e

python prepare_data.py \
  --aasce-root /path/to/boostnet_labeldata \
  --spinalai-root /path/to/Spinal-AI2024 \
  --workspace /path/to/private-workspace
```

Preparation checks the image hashes, rebuilds labels locally and keeps held-out labels in a separate file. It refuses to overwrite existing prepared inputs. Do not commit the workspace. No image is downloaded, uploaded or transmitted by the preparation command.

## Run on your server

```bash
python run.py preflight --workspace /path/to/private-workspace --weights /path/to/dinov2_vits14_pretrain.pth
python run.py train --workspace /path/to/private-workspace --weights /path/to/dinov2_vits14_pretrain.pth --job-id C1-SOURCE-PRETRAIN-CE
python run.py train --workspace /path/to/private-workspace --weights /path/to/dinov2_vits14_pretrain.pth --job-id C1-C1_R1_REAL_ONLY_CE-B10-S2026081901
```

`python run.py plan` lists the source job followed by the 60 target jobs. Execute each target job separately or schedule that list on your server. All staged fits reuse the one source checkpoint. Training runs the full epoch budget; validation selects the target checkpoint but does not stop training early. The runner supports resuming its own local checkpoints. Only load checkpoints created in a trusted local workspace, because resume files contain Python-serialized optimizer state.

Once all planned training jobs are complete, evaluate one target job with:

```bash
python run.py evaluate --workspace /path/to/private-workspace --weights /path/to/dinov2_vits14_pretrain.pth --job-id C1-C1_R1_REAL_ONLY_CE-B10-S2026081901
```

Evaluation writes only to the private workspace and refuses to overwrite a previous evaluation folder. Nothing is automatically pushed to GitHub.

## Implementation provenance and limits

`engine/` contains function bodies extracted from the existing implementation. `CODE_PROVENANCE.json` identifies their source module and SHA-256. `prepare_data.py` and `run.py` are publication-specific adapters: they replace machine-specific paths and historical experiment-lock dependencies with a fresh local input/code fingerprint. They do not claim to reproduce historical execution locks. Original experiment artifacts remain outside this repository.

This is an operational three-band classification task based on the maximum supplied regional Cobb angle: below 30 degrees, 30 to below 45 degrees, and at least 45 degrees. These bands are not a validated clinical triage policy. Containment groups are similarity-based units, not verified patient identifiers. This repository makes no external clinical validation or diagnostic safety claim.

## Rights and attribution

Third-party datasets, pretrained weights and dependencies retain their original ownership and terms. This release does not relicense them. A software licence for the project-specific code has not yet been selected by the rights holder; public visibility alone should not be interpreted as an additional licence grant. See [NOTICE.md](NOTICE.md).
