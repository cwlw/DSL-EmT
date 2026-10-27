import os
import pickle
import numpy as np
from tqdm import tqdm

# 你的 DEAP 输出路径
DATA_ROOT = "/root/autodl-tmp/DEAP_out/compacted_48"
INFO_PATH = "/root/autodl-tmp/DEAP_out/dataset_info.pkl"

def main():
    with open(INFO_PATH, 'rb') as f:
        dataset_info = pickle.load(f)

    trial_list = dataset_info['trial']
    
    rp_min = np.inf
    rp_max = -np.inf
    de_sum = 0.0
    de_sum_sq = 0.0
    de_count = 0

    print("Calculating global stats for DEAP...")
    for trial in tqdm(trial_list):
        trial_dir = os.path.join(DATA_ROOT, trial)
        rp_path = os.path.join(trial_dir, 'eeg_RP.npy')
        de_path = os.path.join(trial_dir, 'eeg_DE.npy')

        rp = np.load(rp_path)
        de = np.load(de_path)

        rp_min = min(rp_min, np.min(rp))
        rp_max = max(rp_max, np.max(rp))
        
        de_sum += np.sum(de)
        de_sum_sq += np.sum(de ** 2)
        de_count += de.size

    de_mean = de_sum / de_count
    de_var = (de_sum_sq / de_count) - (de_mean ** 2)
    de_std = np.sqrt(max(de_var, 0.0))

    dataset_info['feature_stats'] = {
        'eeg_RP': {'min': float(rp_min), 'max': float(rp_max)},
        'eeg_DE': {'mean': float(de_mean), 'std': float(de_std)}
    }

    with open(INFO_PATH, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print("Stats calculated and saved successfully!")
    print(f"rPSD: min={rp_min:.4f}, max={rp_max:.4f}")
    print(f"DE: mean={de_mean:.4f}, std={de_std:.4f}")

if __name__ == '__main__':
    main()