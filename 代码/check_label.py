import scipy.io as sio
import numpy as np
import os

# 路径基于你之前提供的路径推导
mat_path = "/root/autodl-tmp/MAHNOB/lable_continous_Mahnob.mat" 

if not os.path.exists(mat_path):
    print(f"找不到文件: {mat_path}")
    print("请确认 configs.py 里的 root_directory 路径，并用 ls 检查文件是否存在。")
else:
    mat = sio.loadmat(mat_path)
    labels = np.squeeze(mat['labels'])

    print(f"总 trial 数: {len(labels)}")
    print(f"第 0 个 trial 转置前的 Shape: {labels[0].shape}")
    print(f"第 0 个 trial 转置后的 Shape: {labels[0].T.shape}")
    print(f"第 0 个 trial 转置后的前 3 行数据:\n {labels[0].T[:3]}")
