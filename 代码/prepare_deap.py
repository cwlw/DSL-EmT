import os
import pickle
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

# ================= 配置参数 =================
DEAP_RAW_PATH = "/root/autodl-tmp/DEAP/data_preprocessed_python"
OUTPUT_PATH = "/root/autodl-tmp/DEAP_out/compacted_48" # 输出路径
SAMPLING_RATE = 128
WINDOW_SEC = 2
HOP_SEC = 0.25
BANDS = [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]
NUM_CHAN = 32

os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_features(eeg_data):
    """eeg_data: (32, 7680) 60s数据"""
    win_len = int(WINDOW_SEC * SAMPLING_RATE) # 256
    hop_len = int(HOP_SEC * SAMPLING_RATE)    # 32
    num_frames = (eeg_data.shape[1] - win_len) // hop_len + 1
    
    rpsd_list, de_list = [], []
    
    for i in range(num_frames):
        start = i * hop_len
        end = start + win_len
        segment = eeg_data[:, start:end] # (32, 256)
        
        freqs, psd = welch(segment, fs=SAMPLING_RATE, nperseg=min(256, win_len))
        
        frame_rpsd, frame_de = [], []
        for b_low, b_high in BANDS:
            idx = np.where((freqs >= b_low) & (freqs <= b_high))[0]
            band_psd = np.trapz(psd[:, idx], freqs[idx], axis=1) if len(idx) > 0 else np.zeros(segment.shape[0])
            
            # DE 计算
            variance = np.var(segment, axis=1)
            band_de = 0.5 * np.log2(2 * np.pi * np.e * (variance + 1e-6))
            
            frame_rpsd.append(band_psd)
            frame_de.append(band_de)
            
        rpsd_list.append(np.stack(frame_rpsd, axis=1).flatten())
        de_list.append(np.stack(frame_de, axis=1).flatten())
        
    return np.array(rpsd_list), np.array(de_list)

def main():
    dataset_info = {
        "trial": [], "subject_no": [], "trial_no": [],
        "length": [], "has_continuous_label": [], "has_eeg": [],
        "multiplier": {"eeg_RP": 1, "eeg_DE": 1},
        "data_folder": "compacted_48",
        "feature_dimension": {"eeg_RP": (32, 6), "eeg_DE": (32, 6), "continuous_label": (1,)},
    }
    
    for sub_id in tqdm(range(1, 33), desc="Processing Subjects"):
        dat_file = os.path.join(DEAP_RAW_PATH, f"s{sub_id:02d}.dat")
        if not os.path.exists(dat_file):
            continue
            
        with open(dat_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
            
        data = data_dict['data']    # (40, 40, 8064)
        labels = data_dict['labels'] # (40, 4)
        
        for trial_idx in range(40):
            eeg = data[trial_idx, :NUM_CHAN, 384:] # 丢弃前3秒基线
            
            val_score = labels[trial_idx, 0] 
            val_norm = (val_score - 1) / 8.0 
            
            rpsd_feat, de_feat = extract_features(eeg) 
            T_len = rpsd_feat.shape[0]
            
            continuous_label = np.full((T_len, 1), val_norm, dtype=np.float32)
            
            trial_name = f"s{sub_id:02d}_t{trial_idx:02d}"
            trial_dir = os.path.join(OUTPUT_PATH, trial_name)
            os.makedirs(trial_dir, exist_ok=True)
            
            np.save(os.path.join(trial_dir, "eeg_RP.npy"), rpsd_feat)
            np.save(os.path.join(trial_dir, "eeg_DE.npy"), de_feat)
            np.save(os.path.join(trial_dir, "continuous_label.npy"), continuous_label)
            
            dataset_info["trial"].append(trial_name)
            dataset_info["subject_no"].append(sub_id)
            dataset_info["trial_no"].append(trial_idx)
            dataset_info["length"].append(T_len)
            dataset_info["has_continuous_label"].append(1)
            dataset_info["has_eeg"].append(1)
            
    info_path = os.path.join(os.path.dirname(OUTPUT_PATH), "dataset_info.pkl")
    with open(info_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    print(f"DEAP 特征提取完成！")

if __name__ == '__main__':
    main()