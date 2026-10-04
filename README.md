# SEM Microstructure Segmentation and Particle Size Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project segments precipitate particles in Scanning Electron Microscopy (SEM) images and measures their size distribution. It uses a Residual Attention U-Net to segment particles, marker-controlled watershed to split touching particles, and calculates area-weighted particle size distributions (PSD).

---

## Architecture

![Residual Attention U-Net Architecture](figures/unet_architecture.png)

The network is a U-Net with attention gates and residual blocks:
* **Encoder:** 4 downsampling stages (16, 32, 64, 128 channels) with $2 \times 2$ max pooling.
* **Residual Blocks:** Each block has two $3 \times 3$ convolutions with Batch Normalization, ReLU, and a skip connection.
* **Attention Gates:** Placed on skip connections to reduce background noise from scratches and matrix contrast.
* **Deep Supervision:** Auxiliary outputs at intermediate decoder levels to help training converge.
* **Parameters:** ~2.06 million trainable parameters.

---

## Validation Results

Evaluated on 23 validation images with 50% sliding-window overlap:

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Dice Coefficient** | **0.842 ± 0.071** | Peak: 0.8756 |
| **IoU (Jaccard Index)** | **0.733 ± 0.088** | Peak: 0.7835 |
| **Precision** | **0.864 ± 0.065** | Correctly identified particle pixels |
| **Recall** | **0.822 ± 0.078** | Fraction of reference particles detected |
| **Pixel Accuracy** | **97.1 ± 1.2%** | Overall pixel classification accuracy |
| **Predicted Area Fraction** | **10.97 ± 2.8%** | Reference area fraction: 11.73 ± 3.1% |

---

## Repository Structure

```text
Microstructure-SEM-image-Analysis/
├── configs/
│   ├── config.yaml              # Training, loss, and watershed settings
│   └── calibration.yaml         # Scale calibration (nm/pixel) by magnification
├── checkpoints/
│   └── best_model.pth           # Trained model weights
├── figures/
│   ├── unet_architecture.png    # Architecture diagram (PNG)
│   └── unet_architecture.pdf    # Architecture diagram (PDF)
├── src/
│   ├── data/                    # Patch extraction, inpainting, and dataset loaders
│   ├── models/                  # U-Net, residual blocks, and attention gates
│   ├── losses/                  # Dice loss, BCE loss, and metrics
│   ├── engine/                  # Training loop and sliding-window inference
│   ├── postprocess/             # Morphological cleaning and watershed splitting
│   └── analysis/                # Particle sizing, area fraction, and PSD calculation
├── scripts/
│   ├── 01_prepare_data.py       # Extract patches from raw images
│   ├── 02_train.py              # Train the model
│   ├── 03_predict.py            # Run inference on full images
│   ├── 04_run_watershed.py      # Split touching particles
│   └── 05_analyze_microstructure.py # Sizing and area-weighted PSD plots
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
└── README.md
```

---

## How to Use

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/saishivaranjan92-dot/Microstructure-SEM-image-Analysis.git
cd Microstructure-SEM-image-Analysis
pip install -r requirements.txt
```

### 2. Prepare Data
Put raw SEM images in `data/raw/` and masks in `data/masks/`:
```bash
python scripts/01_prepare_data.py --config configs/config.yaml
```

### 3. Train the Model
```bash
python scripts/02_train.py --config configs/config.yaml
```

### 4. Run Prediction
Run sliding-window inference on SEM images using the trained weights:
```bash
python scripts/03_predict.py --input data/raw/ --weights checkpoints/best_model.pth
```
This saves probability maps and binary masks (threshold 0.50) to `outputs/`.

### 5. Split Touching Particles
Separate touching particles using marker-controlled watershed:
```bash
python scripts/04_run_watershed.py --input outputs/binary_masks/
```

### 6. Calculate Particle Sizes and PSD
Calculate equivalent circular diameter ($d_{\text{eq}}$), area fraction, and 25 nm area-weighted PSD with Gaussian Mixture Model (GMM) fitting:
```bash
python scripts/05_analyze_microstructure.py --instances outputs/watershed_instances/ --mag 3000x
```
This outputs summary CSV files and PSD plots to `outputs/psd_results/`.

---

## Particle Size Distribution (PSD) Details

* **Area-Weighted Sizing:**
  Rather than only counting particle numbers, each 25 nm size bin is weighted by total particle area:
  $$Y_k = \sum_{i \in \text{bin } k} A_i \quad [\mu\text{m}^2]$$
  This gives a clearer picture of how particles of different sizes contribute to the total precipitate volume.

* **Edge Particles:**
  Particles touching the image border are kept when measuring total area fraction, but excluded from size and shape distributions because they are cut off by the border.

---

## Pixel Scales

| Magnification | Scale (nm/px) | Field of View |
| :---: | :---: | :---: |
| 2,000× | 49.60 | ~63.5 × 47.6 µm |
| 3,000× | 33.07 | ~42.3 × 31.7 µm |
| 5,000× | 19.84 | ~25.4 × 19.0 µm |
| 20,000× | 4.96 | ~6.35 × 4.76 µm |

---

## License
MIT License. See [LICENSE](LICENSE) for details.
