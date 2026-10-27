import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse


def csv_generator(path_to_read_result, num_folds=24, **kwargs):
    """
    汇总所有 LOSO fold 的 training_logs.csv，
    输出 validation / test 的 RMSE、PCC、CCC 均值和标准差。

    兼容当前修改后的 training_logs.csv：
    - args/config 信息
    - epoch 训练日志
    - feature_mode
    - fusion_alpha
    """

    fold_wise_csv_filename = "training_logs.csv"
    csv_filename = os.path.join(path_to_read_result, "result.csv")

    metrics = ["rmse", "pcc", "ccc"]

    validate_result = np.zeros((len(metrics), num_folds + 2), dtype=np.float64)
    test_result = np.zeros((len(metrics), num_folds + 2), dtype=np.float64)

    # 用于确认已经成功读取了哪些 fold
    processed_folds = []

    for path in Path(path_to_read_result).rglob(fold_wise_csv_filename):

        # ------------------------------------------------------------
        # 1. 获取 fold 编号
        # ------------------------------------------------------------
        try:
            fold = int(path.parts[-2].split('_')[-2])
        except (ValueError, IndexError) as exc:
            print(f"[WARNING] 无法解析 fold 编号，跳过：{path}")
            print(f"原因：{exc}")
            continue

        if fold < 0 or fold >= num_folds:
            print(f"[WARNING] fold={fold} 超出范围 [0, {num_folds - 1}]，跳过：{path}")
            continue

        print(f"Processing fold {fold}: {path}")

        # ------------------------------------------------------------
        # 2. 当前 training_logs.csv 前面包含 args/config 信息。
        #    原代码 skiprows=4 的方式保留，但增加鲁棒性检查。
        # ------------------------------------------------------------
        try:
            df = pd.read_csv(path, skiprows=4)
        except Exception as exc:
            print(f"[ERROR] 无法读取：{path}")
            print(f"原因：{exc}")
            continue

        # ------------------------------------------------------------
        # 3. 检查必要字段
        # ------------------------------------------------------------
        required_columns = [
            "epoch",
            "best_epoch",
            "val_rmse",
            "val_pcc_v",
            "val_ccc"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(
                f"[ERROR] fold {fold} 缺少字段：{missing_columns}"
            )
            print(f"当前字段：{list(df.columns)}")
            continue

        # ------------------------------------------------------------
        # 4. 将 epoch / best_epoch 显式转换为数值
        #    这是修复当前 IndexError 的关键。
        # ------------------------------------------------------------
        df["epoch"] = pd.to_numeric(
            df["epoch"],
            errors="coerce"
        )

        df["best_epoch"] = pd.to_numeric(
            df["best_epoch"],
            errors="coerce"
        )

        # 删除无法解析的日志行
        epoch_df = df.dropna(
            subset=[
                "epoch",
                "best_epoch",
                "val_rmse",
                "val_pcc_v",
                "val_ccc"
            ]
        ).copy()

        if epoch_df.empty:
            print(f"[ERROR] fold {fold} 没有有效 epoch 训练记录")
            continue

        # ------------------------------------------------------------
        # 5. 找到该 fold 最终记录中的 best_epoch
        #
        # 原代码：
        # best_epoch = str(...)
        #
        # 这里绝不能转成 str。
        # ------------------------------------------------------------
        best_epoch = int(
            epoch_df["best_epoch"].iloc[-1]
        )

        # ------------------------------------------------------------
        # 6. 找到 best_epoch 对应的验证结果
        # ------------------------------------------------------------
        result_of_best_epoch = epoch_df.loc[
            epoch_df["epoch"] == best_epoch
        ]

        # 如果因为浮点表示等原因没有直接匹配，
        # 再使用最接近的 epoch 做兜底。
        if result_of_best_epoch.empty:

            epoch_difference = (
                np.abs(epoch_df["epoch"] - best_epoch)
            )

            closest_index = epoch_difference.idxmin()

            result_of_best_epoch = epoch_df.loc[
                [closest_index]
            ]

            print(
                f"[WARNING] fold {fold}: "
                f"无法直接匹配 best_epoch={best_epoch}，"
                f"已使用最近 epoch={epoch_df.loc[closest_index, 'epoch']}"
            )

        # ------------------------------------------------------------
        # 7. 读取 validation 指标
        # ------------------------------------------------------------
        best_row = result_of_best_epoch.iloc[0]

        val_rmse = float(best_row["val_rmse"])
        val_pcc = float(best_row["val_pcc_v"])
        val_ccc = float(best_row["val_ccc"])

        # ------------------------------------------------------------
        # 8. 读取最后一行测试结果
        #
        # 当前日志结构中，训练 epoch 行之后，
        # 最后一行是 Test results。
        # 原代码的 iloc[2]/iloc[4]/iloc[7] 保留。
        # ------------------------------------------------------------
        last_row = df.iloc[-1, :]

        try:
            test_rmse = float(last_row.iloc[2])
            test_pcc = float(last_row.iloc[4])
            test_ccc = float(last_row.iloc[7])
        except (ValueError, TypeError, IndexError) as exc:

            print(
                f"[ERROR] fold {fold} 测试结果读取失败：{exc}"
            )

            print(
                f"最后一行内容：\n{last_row}"
            )

            continue

        # ------------------------------------------------------------
        # 9. 写入结果矩阵
        # ------------------------------------------------------------
        test_result[0, fold] = test_rmse
        test_result[1, fold] = test_pcc
        test_result[2, fold] = test_ccc

        validate_result[0, fold] = val_rmse
        validate_result[1, fold] = val_pcc
        validate_result[2, fold] = val_ccc

        processed_folds.append(fold)

        # 当前实验特征模式
        feature_mode = "unknown"
        if "feature_mode" in df.columns:
            non_null_modes = df["feature_mode"].dropna()
            if len(non_null_modes) > 0:
                feature_mode = str(non_null_modes.iloc[-1])

        print(
            f"fold={fold} | "
            f"feature_mode={feature_mode} | "
            f"best_epoch={best_epoch} | "
            f"val RMSE={val_rmse:.6f} | "
            f"val PCC={val_pcc:.6f} | "
            f"val CCC={val_ccc:.6f} | "
            f"test RMSE={test_rmse:.6f} | "
            f"test PCC={test_pcc:.6f} | "
            f"test CCC={test_ccc:.6f}"
        )

    # ------------------------------------------------------------
    # 10. 检查是否有 fold 没有结果
    # ------------------------------------------------------------
    processed_folds = sorted(set(processed_folds))

    missing_folds = [
        fold
        for fold in range(num_folds)
        if fold not in processed_folds
    ]

    if missing_folds:
        print(
            f"[WARNING] 以下 fold 没有成功汇总：{missing_folds}"
        )

    if not processed_folds:
        raise RuntimeError(
            "没有成功读取任何 fold 的训练结果，"
            "请检查 save 路径和 training_logs.csv 格式。"
        )

    # ------------------------------------------------------------
    # 11. 只对实际成功读取的 fold 计算 mean/std
    # ------------------------------------------------------------
    valid_fold_indices = np.array(
        processed_folds,
        dtype=np.int64
    )

    test_result[:, -2] = np.mean(
        test_result[:, valid_fold_indices],
        axis=1
    )

    test_result[:, -1] = np.std(
        test_result[:, valid_fold_indices],
        axis=1
    )

    validate_result[:, -2] = np.mean(
        validate_result[:, valid_fold_indices],
        axis=1
    )

    validate_result[:, -1] = np.std(
        validate_result[:, valid_fold_indices],
        axis=1
    )

    # ------------------------------------------------------------
    # 12. 输出 validation result
    # ------------------------------------------------------------
    columns = [
        str(i) for i in range(num_folds)
    ] + ["mean", "std"]

    result_csv = pd.DataFrame(
        validate_result,
        columns=columns,
        index=metrics
    ).rename_axis("val")

    result_csv.to_csv(
        csv_filename,
        mode="w"
    )

    # ------------------------------------------------------------
    # 13. 输出 test result
    # ------------------------------------------------------------
    result_csv = pd.DataFrame(
        test_result,
        columns=columns,
        index=metrics
    ).rename_axis("test")

    result_csv.to_csv(
        csv_filename,
        mode="a"
    )

    print()
    print("=" * 80)
    print("Result summary completed.")
    print(f"Processed folds: {processed_folds}")
    print(f"Result file: {csv_filename}")
    print("=" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate LOSO regression result CSV."
    )

    parser.add_argument(
        "-path_to_read_result",
        default=r"/root/autodl-tmp/EMT论文打包/EmT-main/regression/save",
        type=str,
        help="The root directory containing fold result folders."
    )

    args = parser.parse_args()

    csv_generator(
        args.path_to_read_result,
        num_folds=24
    )