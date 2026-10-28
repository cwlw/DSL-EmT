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

```bash
# Create virtual environment
conda create -n dslemt python=3.10.8
conda activate dslemt

# Install dependencies via pip
pip install -r requirements.txt
```

*Note: Please refer to the `requirements.txt` file in the repository for specific library versions, including `einops`, `tqdm`, `scikit-learn`, and `pandas`.*

## 📁 Repository Structure

```text
.
├── base/                       # Core functional modules
│   ├── checkpointer.py         # Model saving and checkpoint recovery
│   ├── dataset.py              # Data loading, windowing, and LOSO partitioning
│   ├── egg.py                  # EEG signal processing helpers (Note: filename reflects repo state)
│   ├── experiment.py           # Base experiment class defining general workflows
│   ├── feature_fusion.py       # Early feature fusion module (replaced by internal α gate in EmT)
│   ├── label_configs.py        # Label dimension mapping configurations
│   ├── logger.py               # Metric calculation (RMSE, PCC, CCC) and logging
│   ├── loss_function.py        # Hybrid losses (CCC, MSE, Temporal Difference, Smoothness)
│   ├── parameter_control.py    # Optimizer and learning rate control
│   ├── preprocessing.py        # Raw EEG filtering and preprocessing base class
│   ├── scheduler.py            # Learning rate scheduler (Warmup + ReduceLROnPlateau)
│   ├── trainer.py              # Training/testing loops with channel masking logic
│   └── utils.py                # General utilities (pickle IO, file management)
├── model/                      # Model architecture
│   └── EmT.py                  # Core network: Dual-stream RMPG, learnable α gate, TCT, 1-layer Bi-GRU
├── main.py                     # Main entry point, parses CLI args and starts experiments
├── experiment.py               # Inherits from base, assembles Model, Dataset, and Trainer
├── configs.py                  # Global configuration (hyperparameters, paths, frequency bands)
├── generate_dataset.py         # Offline preprocessing script for MAHNOB-HCI
├── generate_result_csv.py      # Script to aggregate MAHNOB 24-fold LOSO results
│
├── prepare_deap.py             # [DEAP Extension] Feature extraction script for DEAP
├── fix_deap_stats.py           # [DEAP Extension] Calculates global normalization stats for DEAP
├── calc_deap_acc.py            # [DEAP Extension] Calculates Trial-level binary accuracy on DEAP
├── test_deap_only.py           # [DEAP Extension] Pure inference script for quick DEAP validation
├── check_label.py              # Helper script to inspect label shapes and distributions
└── README.md                   # Project documentation
```

## ⚠️ Guide for DEAP Extension

**Important**: The core scripts in this repository (e.g., `dataset.py`, `trainer.py`, `experiment.py`) are configured by default for **MAHNOB-HCI (Continuous Trajectory Regression)**. 

Because DEAP provides only a single discrete valence rating per trial, it requires a `Sequence-to-One` regression protocol. To reproduce the DEAP extension experiments in the paper, you need to make the following modifications:

1. **`configs.py`**:
   * Change `"sampling_frequency"` in `eeg_config` from `256` to `128`.
   * Update `"output_root_directory"` to point to your DEAP output path.
2. **`base/dataset.py`**:
   * Modify `DataArranger.partition_range_fn()`: Change the hardcoded 24-fold division to dynamically generate 32 folds based on `subject_no`.
   * Modify `_EEGFeatureMixin.get_example()`: For continuous labels, use `np.mean(example, axis=0)` to average them over the time dimension, producing a single value `(1,)`.
3. **`model/EmT.py`**:
   * Modify the `forward()` function: After `self.transformer`, add `fused_pooled = torch.mean(fused, dim=1)` to pool the 96-frame sequence into a single feature before feeding it into `self.MLP`.
4. **`base/trainer.py`**:
   * Modify the `loop()` function: Remove the label dimension check and zeroing. After `outputs = self.model(...)`, expand the dimensions of both `outputs` and `labels` (`unsqueeze(1).repeat(1, seq_len, 1)`) to adapt to the 3D metric calculation in `logger.py`.
5. **`generate_result_csv.py`**:
   * Change `num_folds=24` to `num_folds=32` to aggregate DEAP results.

*(Note: When running DEAP training, you **must** add the parameter `-use_temporal 0` to disable the temporal difference loss, as trial-level labels do not have adjacent-frame variations).*

## 🚀 Training & Evaluation

### 1. Data Preparation

* **MAHNOB-HCI**: Follow the EmT benchmark protocol. Place your preprocessed `.npy` features in a directory and update the paths in `configs.py`.
* **DEAP**: Use the `prepare_deap.py` and `fix_deap_stats.py` scripts to generate features and stats.

### 2. Reproducing MAHNOB-HCI Main Results (Table I)

To run the 24-fold LOSO cross-validation for the full DSL-EmT model:

```bash
python main.py \
  -experiment_name DSL_EmT_MAHNOB \
  -stamp MAHNOB_A0 \
  -gpu 0 \
  -case loso \
  -task reg \
  -num-eeg-chan 32 \
  -num-f 6 \
  -window_length 96 \
  -hop_length 32 \
  -batch_size 2 \
  -mse_weight 2.0 \
  -use_temporal 1 \
  -temporal_weight 0.05 \
  -dataset_path [YOUR_MAHNOB_OUT_PATH]
```

### 3. Reproducing DEAP Extension Results

*(Remember to follow the "Guide for DEAP Extension" above and disable temporal loss)*

```bash
python main.py \
  -experiment_name DSL_EmT_DEAP \
  -stamp DEAP_TrialReg \
  -gpu 0 -case loso -task reg \
  -num-eeg-chan 32 -num-f 6 \
  -window_length 96 -hop_length 32 \
  -batch_size 2 \
  -mse_weight 2.0 \
  -use_temporal 0 \
  -dataset_path [YOUR_DEAP_OUT_PATH]
```

### 4. Aggregating Results

After training, use the `generate_result_csv.py` script to aggregate the fold results into a single `result.csv`:

```bash
python generate_result_csv.py -path_to_read_result [YOUR_SAVE_PATH]
```

## 📦 Pre-trained Models & Weights

We provide the weights and training logs for the full DSL-EmT model (MAHNOB-HCI, all 24 LOSO folds).

* **Baidu Netdisk**: [https://pan.baidu.com/s/1MBAx0onlvE8oyNH9Qlbj6w?pwd=pp2s](https://pan.baidu.com/s/1MBAx0onlvE8oyNH9Qlbj6w?pwd=pp2s) 
* **Extraction Code**: `pp2s`

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{Author2025dslemt,
  title={DSL-EmT: A Dual-Stream Lightweight Temporal Difference Transformer for Cross-Subject EEG Valence Regression},
  author={Author 1, Author 2, Author 3},
  journal={Under Review},
  year={2025}
}
```

## 🙏 Acknowledgements

This codebase is heavily built upon the open-source [EmT repository](https://github.com/yi-ding-cs/EmT). We thank the authors for their excellent work.

## 📬 Contact

For any questions or bug reports, please contact: 3327264882@qq.com

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
