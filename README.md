# PAR-SegNet: Pairwise Anatomical Relation Modeling Network for Fetal Head and Pubic Symphysis Segmentation in Intrapartum Ultrasound

PAR-SegNet is a U-Net-based framework for joint segmentation of the pubic symphysis (PS) and fetal head (FH) in intrapartum ultrasound (IUS).

The network is centered on **Pairwise Anatomical Relation Modeling (PARM)**, which learns feature-level relationships between PS-oriented and FH-oriented representations and uses the learned relations to refine the two representations through separate paths. Enhanced Feature Recalibration (EFR) and Adaptive Multi-scale Context Aggregation (AMCA) are incorporated to improve feature reliability and multi-scale contextual representation.

## Overview

Accurate segmentation of the pubic symphysis and fetal head is important for estimating the Angle of Progression (AoP), a quantitative measure used in the assessment of labor progression. Intrapartum ultrasound presents several challenges, including speckle interference, weak anatomical boundaries, and substantial scale differences between the two structures.

PAR-SegNet addresses these challenges through three components:

- **Enhanced Feature Recalibration (EFR):** Applies lightweight SE-style channel recalibration to improve feature responses across hierarchical network stages.
- **Adaptive Multi-scale Context Aggregation (AMCA):** Aggregates complementary contextual features using parallel depthwise convolutions with different receptive fields and spatially adaptive scale weights.
- **Pairwise Anatomical Relation Modeling (PARM):** Constructs feature-level pairwise relations from discrepancies, interactions, and joint responses between two independently parameterized feature representations, followed by relation-guided asymmetric refinement.

The network adopts a U-Net-style encoder-decoder architecture with skip connections. AMCA and PARM are placed at the bottleneck, while EFR is integrated into the convolutional blocks.

## Repository Structure

```text
PAR-SegNet/
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── evaluate.py
├── ellipse.py
├── models/
│   ├── __init__.py
│   ├── par_segnet.py
│   ├── efr.py
│   ├── amca.py
│   └── parm.py
├── datasets/
│   ├── __init__.py
│   ├── dataset.py
│   └── README.md
└── utils/
    ├── __init__.py
    └── losses.py
```

- `models/`: Model architecture and the EFR, AMCA, and PARM modules.
- `datasets/`: Dataset loading and joint image-mask preprocessing.
- `utils/losses.py`: Dice-based loss functions.
- `train.py`: Model training.
- `predict.py`: Inference and saving predicted segmentation masks.
- `evaluate.py`: Segmentation and downstream AoP evaluation.
- `ellipse.py`: Geometric utility used for AoP calculation.
- `requirements.txt`: Python package dependencies.

## Datasets

The datasets used in this study are publicly available through Zenodo.

### 1. MICCAI 2023 FH-PS-AOP Challenge Training Set

- Number of images: 4,000.
- Purpose: Model development and validation.
- Image format: MetaImage (`.mha`).
- Image dimensions: 3 × 256 × 256.
- Label dimensions: 256 × 256.

Dataset: [FH-PS-AOP Challenge Training Set — Zenodo](https://doi.org/10.5281/zenodo.7851339)

### 2. PSFHS Dataset

- Number of annotated ultrasound images: 1,358.
- Purpose: Separate cross-dataset evaluation.
- Use in this study: The dataset was not used for model training or validation.

Dataset: [PSFHS — Zenodo](https://doi.org/10.5281/zenodo.10969427)

Please download both datasets from their original repositories and follow the applicable dataset terms. The datasets themselves are not included in this GitHub repository.

### Label Convention

The segmentation masks use the following class IDs:

| Class ID | Anatomical structure |
|---|---|
| 0 | Background |
| 1 | Pubic symphysis (PS) |
| 2 | Fetal head (FH) |

Ensure that the label mapping in your local dataset matches this convention before training or evaluation.

### Dataset Organization

The current training dataset loader expects a directory structure similar to the following:

```text
dataset_root/
├── split/
│   ├── train.txt
│   └── val.txt
└── train/
    ├── images/
    │   ├── <case_id>.mha
    │   └── ...
    └── masks/
        ├── <case_id>.mha
        └── ...
```

The split files contain image filename stems without the `.mha` extension. Each image must have a matching segmentation mask.

The prediction script also uses test-image and test-label paths that should be configured for the local PSFHS data organization. See `datasets/README.md` and the path settings in the scripts before running the experiments.

## Environment

The experiments reported in the manuscript were conducted using:

- **PyTorch:** 1.13.1
- **GPU:** NVIDIA GeForce RTX 4090
- **GPU memory:** 24 GB
- **Input resolution:** 256 × 256
- **Batch size:** 8

The packages required by the current implementation are listed in `requirements.txt`.

### Installation

Clone the repository:

```bash
git clone https://github.com/JSKKG/PAR-SegNet.git
cd PAR-SegNet
```

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For closer reproduction of the reported experiments, use PyTorch 1.13.1 with a compatible torchvision and CUDA configuration. Check the installed versions and your local GPU environment before training.

## Training Configuration

The experimental protocol described in the manuscript is summarized below.

| Setting | Value |
|---|---|
| Development dataset | FH-PS-AOP Challenge training set |
| Number of randomized splits | 5 |
| Training images per split | 3,200 |
| Validation images per split | 800 |
| Maximum training epochs | 60 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 3 × 10⁻⁴ |
| Weight decay | 3 × 10⁻⁵ |
| Learning-rate schedule | Polynomial decay, power 0.9 |
| Training objective | 0.5 × CE loss + 0.5 × DC loss |
| Model selection | Lowest mean validation Dice loss among evaluated checkpoints |

Validation was performed from epoch 35 onward, and the selected checkpoint was evaluated on the separate PSFHS dataset.

The training implementation and configuration should be checked against the settings above before reproducing the reported results.

## Usage

The repository provides separate scripts for training, inference, and evaluation. The current scripts retain path settings from the original research environment; update those paths to match your local dataset and checkpoint locations before execution.

### 1. Initialize the Model

```python
from models.par_segnet import PARSegNet

model = PARSegNet(in_ch=3, out_ch=3)
```

The model produces three-channel segmentation logits corresponding to background, PS, and FH.

### 2. Train the Model

After configuring the dataset paths and training settings in `train.py`, run:

```bash
python train.py
```

### 3. Generate Predictions

Configure the input images, labels, and trained checkpoint paths in `predict.py`, then run:

```bash
python predict.py
```

The prediction script saves class-index segmentation masks for subsequent evaluation.

### 4. Evaluate Predictions

After generating prediction masks and preparing the corresponding ground-truth masks, configure the input and output paths in `evaluate.py` and run:

```bash
python evaluate.py
```

The evaluation implementation reports segmentation metrics and AoP-related results. Ensure that the mask encoding, image correspondence, and spatial-spacing conventions match the intended evaluation protocol.

**Note:** These commands describe the intended entry points. The scripts may require local path configuration and additional debugging before they can be run unchanged in a new environment.

## Experimental Results

PAR-SegNet was evaluated on the PSFHS dataset using five independent training runs. The results reported in the manuscript are summarized below.

| Metric | PAR-SegNet |
|---|---:|
| PS Dice Similarity Coefficient (DSC) | 86.53 ± 0.42% |
| FH Dice Similarity Coefficient (DSC) | 93.11 ± 0.29% |
| Overall DSC | 92.87 ± 0.17% |
| AoP estimation error (ΔAoP) | 8.02 ± 0.21° |

Results are reported as mean ± standard deviation across five runs. The overall DSC and AoP error summarize segmentation overlap and downstream geometric estimation performance, respectively.

The manuscript additionally reports comparisons with representative CNN-, Transformer-, state-space-model-, and KAN-based segmentation methods, together with module ablations and computational efficiency measurements.
