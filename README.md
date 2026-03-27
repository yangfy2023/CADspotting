# OneFormer3D

OneFormer3D is a unified 3D scene understanding framework for floorplan analysis. This repository contains the official implementation with training and evaluation code.

## Overview

- **Model**: Unified 3D panoptic segmentation and instance detection
- **Dataset**: Floorplan dataset with semantic and instance annotations
- **Framework**: Built on MMDet3D with PTv3 backbone
- **Public Checkpoint**: `best_pq_epoch_1024.pth` available for download

## Quick Start

### 1. Installation

```bash
export PROJECT_ROOT=/path/to/repo
cd "$PROJECT_ROOT"
```

Follow [INSTALL.md](INSTALL.md) to set up the environment:

```bash
conda create -n oneformer3d python=3.10
conda activate oneformer3d
# ... (see INSTALL.md for full steps)
```

### 2. Download Checkpoint

```bash
bash download_public_ckpt.sh
```

### 3. Run Inference on Sample Data

```bash
CONDA_ENV_NAME=oneformer3d bash test.sh
```

This runs inference on the public smoke sample in `data/smoke/`.

## Repository Structure

```
├── oneformer3d/          # Core model code
├── configs/              # Training and evaluation configs
├── tools/                # Data processing and training scripts
├── data/
│   ├── floorplan/        # Public data, used for visualization
│   └── newfloorplan/     # floorplan_without_color, training for our code
├── INSTALL.md            # Detailed installation guide
├── DATASET.md            # Dataset preparation guide
└── README.md             # This file
```

## Data Processing

For custom floorplan data, use the tools in `tools/`:

```bash
# Convert SVG to NPZ format
python tools/preprocess_floorplan.py \
  --datasets train,test,val \
  --raw_base_path data/floorplan \
  --d_samp 0.14285714285714285

# Generate dataset info files
python tools/create_data.py s3dfloorplan \
  --root-path data/newfloorplan \
  --out-dir data/newfloorplan \
  --extra-tag s3dfloorplan
```

See [tools/readme.md](tools/readme.md) for detailed documentation.

## Training & Evaluation

### Training

```bash
CONDA_ENV_NAME=oneformer3d bash train_dist.sh
```

### Evaluation

```bash
python tools/test.py configs/aaai_nocolor_1024_nopooling.py \
  checkpoints/best_pq_epoch_1024.pth
```

## Dataset

The full floorplan dataset is not bundled with this repository. See [DATASET.md](DATASET.md) for:

- Expected dataset layout
- How to prepare your own data
- Info file generation

## Software Stack

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- MMDet3D 1.4.0, MMEngine 0.10.6, MMCV 2.1.0
- spconv-cu121 2.3.8, flash-attn 2.7.4.post1

See [INSTALL.md](INSTALL.md) for complete version specifications.

## Citation

If you use this code, please cite:

```bibtex
@article{oneformer3d,
  title={OneFormer3D: One Transformer for Unified 3D Scene Understanding},
  year={2024}
}

@article{yang2024cadspotting,
  title={Cadspotting: Robust panoptic symbol spotting on large-scale cad drawings},
  author={Yang, Fuyi and Mu, Jiazuo and Zhang, Yanshun and Zhang, Mingqian and Zhang, Junxiong and Luo, Yongjian and Xu, Lan and Yu, Jingyi and Shi, Yujiao and Zhang, Yingliang},
  journal={arXiv preprint arXiv:2412.07377},
  year={2024}
}
```

## License

