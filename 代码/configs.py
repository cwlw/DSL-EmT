import os

emotion = ["Valence"]

config = {

    "extract_class_label": 1,
    "extract_continuous_label": 1,
    "extract_eeg": 1,

    "eeg_folder": "eeg",

    "eeg_config": {
        "sampling_frequency": 128,
        "window_sec": 2,
        "hop_sec": 0.25,
        "buffer_sec": 5,
        "num_electrodes": 32,
        "interest_bands": [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)],
        "f_trans_interest_bands": [(0.1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
        "channel_slice": {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)},
        "features": ["eeg_RP", "eeg_DE"],
        "filter_type": 'cheby2',
        "filter_order": 4
    },

    "save_npy": 1,
    "npy_folder": "compacted_48",
    "dataset_name": "mahnob",
    "emotion_list": emotion,
    "root_directory": r"/root/autodl-tmp/MAHNOB",
    "output_root_directory": r"/root/autodl-tmp/MAHNOB_out",
    "raw_data_folder": "Sessions",

    "multiplier": {
        "video": 16,
        "eeg_raw": 1,
        "eeg_bandpower": 1,
        "eeg_DE": 1,
        "eeg_RP": 1,
        "eeg_Hjorth": 1,
        "continuous_label": 1,
    },

    "feature_dimension": {
        "eeg_raw": (16384,),
        "eeg_bandpower": (32, 6),
        "eeg_DE": (32, 6),
        "eeg_RP": (32, 6),
        "eeg_Hjorth": (96,),
        "continuous_label": (1,),
        "class_label": (1,),
    },

    # ===== 默认 fusion 模式（A0 基线） =====
    "feature_mode": "fusion",

    # ===== MSE 权重 =====
    "mse_weight": 2.0,

    # ===== 通道遮蔽参数（A0 基线：开启，比例 0.1） =====
    "apply_mask": True,
    "mask_ratio": 0.1,

    # ===== 模型内部 α 初始值 =====
    "fusion_alpha_init": 0.7,

    # ===== 时序平滑损失权重 =====
    "smooth_weight": 0.005,

    # ===== 消融用参数 =====
    # fixed_alpha: None 表示可学习（A0/A1/A2/A6/A8/A9）；
    #              0.5 表示固定 α=0.5（A3）；
    #              0.7 表示固定 α=0.7（A4）
    "fixed_alpha": None,

    # gru_layers: 1 表示 1 层双向 GRU（A0）；
    #             2 表示 2 层双向 GRU（A5）
    "gru_layers": 1,

    "max_epoch": 30,
    "min_epoch": 0,
    "model_name": "EmT",
    "backbone": {
        "state_dict": "res50_ir_0.887",
        "mode": "ir",
    },
    "early_stopping": 30,
    "load_best_at_each_epoch": 0,
    "time_delay": 0,
    "metrics": ["rmse", "pcc", "ccc"],
    "save_plot": 0,
}