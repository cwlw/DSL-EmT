# DSL-EmT: A Dual-Stream Lightweight Temporal Difference Transformer for Cross-Subject EEG Valence Regression

This repository is the official PyTorch implementation of the paper:
**"DSL-EmT: A Dual-Stream Lightweight Temporal Difference Transformer for Cross-Subject EEG Valence Regression"** *(Journal/Conference decision pending)*.

## 📖 Introduction

Cross-subject EEG emotion regression is challenging due to high inter-subject variability and the continuous nature of emotional dynamics. We propose **DSL-EmT**, which addresses three practical weaknesses of the existing EmT baseline:
1. **Dual-Stream Encoder**: Encodes rPSD and DE features independently using two Residual Multiview Pyramid GCN (RMPG) streams, fused via a learnable scalar gate \(\alpha\).
2. **Lightweight Temporal Mixer**: Replaces the 2-layer Bi-GRU with a 1-layer Bi-GRU to mitigate overfitting in small-sample LOSO protocols.
3. **Temporal Difference Loss**: Introduces a \(L_{TD}\) constraint to explicitly supervise the trajectory shape (adjacent-frame changes), alongside a smoothness regularizer and training-time channel masking.

## ⚙️ Environment Setup

We recommend using a Conda environment. The code has been tested with Python 3.10.8 and PyTorch 2.1.2.

# Create virtual environment
conda create -n dslemt python=3.10.8
conda activate dslemt

# Install dependencies via pip
pip install -r requirements.txt

Note: Please refer to the requirements.txt file in the repository for specific library versions, including einops, tqdm, scikit-learn, and pandas.


