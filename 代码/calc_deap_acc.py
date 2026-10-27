import os
import numpy as np
import pandas as pd

# ================= 配置路径 =================
SAVE_DIR = "/root/autodl-tmp/EMT论文打包/EmT-main/regression/save"
PREFIX = "DSL_EmT_DEAP_EmT_DEAP_A0_"
DEAP_INFO_PATH = "/root/autodl-tmp/DEAP_out/dataset_info.pkl"

def main():
    # 1. 读取 DEAP 的真实标签
    with open(DEAP_INFO_PATH, 'rb') as f:
        import pickle
        info = pickle.load(f)
    
    # 建立 trial_name -> 真实分数的映射 (去对应的 npy 里读，或者这里直接根据 trial_no 算)
    # 因为 prepare_deap.py 里 val_score = (label - 1) / 8.0
    # 所以 label > 5 相当于 val_score > 0.5
    # 我们直接找预测文件夹里的 txt，然后去原始的 npy 里读真实 label
    
    total_trials = 0
    correct_predictions = 0
    
    print("开始统计 DEAP 二分类准确率 (Valence > 5 为高)...")
    
    for fold in range(32):
        fold_dir = os.path.join(SAVE_DIR, f"{PREFIX}{fold}_valence")
        # 预测结果通常存在 predict/test/valence/ 目录下
        predict_dir = os.path.join(fold_dir, "predict", "test", "valence")
        
        if not os.path.exists(predict_dir):
            print(f"[Warning] Fold {fold} 预测文件夹不存在: {predict_dir}")
            continue
            
        for txt_file in os.listdir(predict_dir):
            if not txt_file.endswith(".txt"):
                continue
                
            trial_name = txt_file.replace(".txt", "")
            
            # 读取模型预测的序列
            preds = pd.read_csv(os.path.join(predict_dir, txt_file), header=None).values.flatten()
            pred_mean = np.mean(preds)
            
            # 读取该 trial 的真实标签 (去原始 npy 读)
            label_path = f"/root/autodl-tmp/DEAP_out/compacted_48/{trial_name}/continuous_label.npy"
            if not os.path.exists(label_path):
                continue
                
            true_labels = np.load(label_path).flatten()
            true_mean = np.mean(true_labels)
            
            # 二分类判定：> 0.5 为正类（High），<= 0.5 为负类（Low）
            pred_class = 1 if pred_mean > 0.5 else 0
            true_class = 1 if true_mean > 0.5 else 0
            
            if pred_class == true_class:
                correct_predictions += 1
                
            total_trials += 1
            
    if total_trials > 0:
        acc = (correct_predictions / total_trials) * 100
        print("=" * 50)
        print(f"DEAP 分类评估完成！")
        print(f"总测试 Trial 数: {total_trials}")
        print(f"预测正确数: {correct_predictions}")
        print(f"🏆 二分类准确率 (Accuracy): {acc:.2f}%")
        print("=" * 50)
    else:
        print("未找到任何预测文件，请检查路径。")

if __name__ == "__main__":
    main()