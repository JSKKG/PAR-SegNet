# Dataset Preparation

This directory contains the dataset loading and preprocessing code for PAR-SegNet. The original ultrasound images and annotations are not included in this repository.

## Dataset Sources

The datasets used in this study are publicly available from Zenodo:

1. **MICCAI 2023 FH-PS-AOP Challenge training set (4,000 cases):** [https://doi.org/10.5281/zenodo.7851339](https://doi.org/10.5281/zenodo.7851339)
2. **PSFHS dataset:** [https://doi.org/10.5281/zenodo.10969427](https://doi.org/10.5281/zenodo.10969427)

Please download the datasets from their original repositories and follow the applicable terms of use. The datasets themselves are not included in this GitHub repository.

## Expected Directory Structure

The current implementation of `ImageToImage2D` expects the dataset directory to follow this structure:

```text
dataset_root/
├── split/
│   ├── train.txt
│   └── val.txt
└── train/
    ├── images/
    │   ├── 00001.mha
    │   └── 00002.mha
    └── masks/
        ├── 00001.mha
        └── 00002.mha
```

Image and mask files must have matching filenames. Each line in a split file should contain a filename stem without the `.mha` suffix, for example:

```text
00001
00002
00003
```

## Image and Label Format

- Images and masks are stored in MetaImage (`.mha`) format.
- Each image must have a corresponding segmentation mask.
- The mask should use integer class IDs for background, pubic symphysis (PS), and fetal head (FH).
- Verify the exact class-ID mapping against the original dataset annotations and the training/evaluation code before running experiments.

## Notes

- The current loader constructs image and mask paths under `train/images/` and `train/masks/` for each split. Check this behavior against your training script before preparing validation data.
- Dataset paths should be configured locally and should not be hard-coded to a specific machine.
- This repository provides the loading code, not the dataset itself.
