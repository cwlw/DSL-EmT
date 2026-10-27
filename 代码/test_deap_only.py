import torch
import os
import numpy as np
import pandas as pd
from base.dataset import DataArranger, MyDatasetPreLoad
from base.utils import load_pickle
from model.EmT import EmT

# ================= 配置 =================
DATASET_PATH = "/root/autodl-tmp/DEAP_out"
SAVE_DIR = "save"
EXPERIMENT_NAME = "DSL_EmT_DEAP_EmT_DEAP_A0_"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_info = load_pickle(os.path.join(DATASET_PATH, "dataset_info.pkl"))
    feature_stats = dataset_info['feature_stats']
    
    # 补全 multiplier 字典，防止 KeyError
    multiplier = dataset_info['multiplier'].copy()
    multiplier['continuous_label'] = 1
    multiplier['eeg_bandpower'] = 1
    
    data_arranger = DataArranger(dataset_info, DATASET_PATH, debug=0, task='reg', case='loso', seed=0)
    
    total_trials = 0
    correct_predictions = 0
    
    print("开始纯推理计算 DEAP 准确率...")
    
    for fold in range(32):
        model_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}{fold}_valence", "model_state_dict.pth")
        if not os.path.exists(model_path):
            print(f"找不到模型文件，跳过: {model_path}")
            continue
            
        # 初始化模型并加载权重
        model = EmT(layers_graph=[1, 2], num_chan=32, num_feature=6, hidden_graph=32, 
                    alpha_init=0.7, single_stream='r', fixed_alpha=None, gru_layers=1).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # 准备测试数据
        data_list = data_arranger.generate_partitioned_trial_list(window_length=96, hop_length=32, fold=fold)
        
        # 注意这里：参数名是 data_list，不是 data
        test_dataset = MyDatasetPreLoad(
            data_list=data_list['test'], 
            continuous_label_dim=0, 
            modality=["eeg_bandpower", "continuous_label"], 
            multiplier=multiplier, 
            feature_dimension=dataset_info['feature_dimension'], 
            window_length=96, 
            mode='test', 
            feature_mode='fusion', 
            feature_stats=feature_stats
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        with torch.no_grad():
            for X, trials, lengths, indices in test_loader:
                rpsd = X['rpsd_feat'].to(DEVICE)
                de = X['de_feat'].to(DEVICE)
                labels = X['continuous_label'].to(DEVICE)
                
                outputs = model(rpsd=rpsd, de=de)
                
                # 逐 trial 评估
                outputs = outputs.cpu().numpy()
                labels = labels.cpu().numpy()
                
                for b in range(outputs.shape[0]):
                    pred_mean = np.mean(outputs[b])
                    true_mean = np.mean(labels[b])
                    
                    # 大于 0.5 判为 High (即原始打分 > 5)
                    pred_class = 1 if pred_mean > 0.5 else 0
                    true_class = 1 if true_mean > 0.5 else 0
                    
                    if pred_class == true_class:
                        correct_predictions += 1
                    total_trials += 1
                    
        print(f"Fold {fold} 处理完成")

    if total_trials > 0:
        print("=" * 50)
        print(f"总测试 Trial 数: {total_trials}")
        print(f"预测正确数: {correct_predictions}")
        print(f"🏆 二分类准确率 (Accuracy): {(correct_predictions / total_trials) * 100:.2f}%")
        print("=" * 50)
    else:
        print("未找到任何模型文件，请检查路径。")

if __name__ == "__main__":
    main()