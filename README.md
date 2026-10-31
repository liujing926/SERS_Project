# Contrastive and Ranking Learning for Prescreening Ag SERS Substrates

This repository contains the official source code for the paper: **"Contrastive and Ranking Learning for Prescreening Ag SERS Substrates toward Rapid Fabrication Optimization"**.

## Repository Structure
- `config/`: Configuration files (`config.yaml`).
- `data/`: Dataset loading scripts (`dataset.py`) and data augmentation (`transforms.py`).
- `models/`: Neural network architectures (`backbone.py`, `ContrastiveLearn.py`, `Supervised_contrastive_loss.py`).
- `script/`: Main execution scripts.
  - `contrast_train.py`: Training script for contrastive learning.
  - `EF_ranking_train.py`: Training script for the ranking model.
  - `test.py`: Script for testing and evaluating model performance.
  - `Grad_CAM.py`: Script for Grad-CAM visualization.
- `utils/`: Utility functions for evaluation, similarity calculation, and image processing.
- `sample_data/`: A small subset of the dataset to demonstrate the code pipeline.
  - `contrastive_data/`: Sample images and metadata for contrastive learning.
  - `ranking_data/`: Sample images and metadata for ranking learning.

## Data Source
The sample images included in sample_data/ are a small subset of the publicly available datasets described in Section 2.1 of the manuscript (e.g., figshare, Illinois Data Bank). They are included here solely to demonstrate the code execution pipeline.

## Dependencies
Ensure you have Python 3.10+ installed. You can install the required packages using:

```bash
pip install -r requirements.txt