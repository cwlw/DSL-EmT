import torch
from torch import nn


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):
        gold = gold.float()
        pred = pred.float()

        if gold.dim() == 3:
            gold = gold.squeeze(-1)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)

        gold_mean = torch.mean(gold, dim=1, keepdim=True)
        pred_mean = torch.mean(pred, dim=1, keepdim=True)

        covariance = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=1, keepdim=True)
        gold_var = torch.var(gold, dim=1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=1, keepdim=True, unbiased=False)

        ccc = 2.0 * covariance / (gold_var + pred_var + torch.pow(gold_mean - pred_mean, 2) + 1e-8)
        ccc_loss = 1.0 - ccc

        if weights is not None:
            ccc_loss = ccc_loss * weights

        out = torch.mean(ccc_loss)
        return out.float()


class HuberLoss(nn.Module):
    """Smooth L1 Loss (Huber) - 比 MSE 对离群值更鲁棒"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, gold):
        pred = pred.float()
        gold = gold.float()
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if gold.dim() == 3:
            gold = gold.squeeze(-1)
        # 使用 torch 自带的 smooth_l1_loss
        return nn.functional.smooth_l1_loss(pred, gold, beta=self.delta, reduction='mean')


class TemporalDifferenceLoss(nn.Module):
    """让预测的变化量与真实变化量一致"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, gold):
        pred = pred.float()
        gold = gold.float()
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if gold.dim() == 3:
            gold = gold.squeeze(-1)

        # 计算相邻时间步的差值
        pred_diff = pred[:, 1:] - pred[:, :-1]
        gold_diff = gold[:, 1:] - gold[:, :-1]

        loss = (pred_diff - gold_diff) ** 2
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class HybridLoss(nn.Module):
    """
    混合损失：支持 CCC + MSE/Huber + Temporal Difference + Smooth
    """
    def __init__(self, mse_weight=0.5, smooth_weight=0.005,
                 use_huber=False, huber_delta=1.0,
                 use_temporal=False, temporal_weight=0.01):
        super().__init__()
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight
        self.use_huber = use_huber
        self.use_temporal = use_temporal
        self.temporal_weight = temporal_weight

        self.ccc_loss = CCCLoss()
        if use_huber:
            self.mse_loss = HuberLoss(delta=huber_delta)
        else:
            self.mse_loss = nn.MSELoss()

        if use_temporal:
            self.temporal_loss = TemporalDifferenceLoss()

    def forward(self, gold, pred, weights=None):
        # 与 CCCLoss 及 Trainer 调用 criterion(labels, outputs) 保持一致：
        # 第一个参数为真值 gold，第二个参数为模型预测 pred。
        gold = gold.float()
        pred = pred.float()

        # 1. CCC损失
        ccc_val = self.ccc_loss(gold, pred, weights)

        # 2. MSE / Huber 损失
        mse_val = self.mse_loss(pred, gold)

        # 3. 时序平滑损失（约束预测 pred 的相邻帧变化，而非真值）
        if pred.dim() == 3:
            pred_smooth = pred.squeeze(-1)
        else:
            pred_smooth = pred
        if pred_smooth.size(1) > 1:
            smooth_loss = torch.mean((pred_smooth[:, 1:] - pred_smooth[:, :-1]) ** 2)
        else:
            smooth_loss = torch.tensor(0.0, device=pred.device)

        # 4. Temporal Difference 损失（预测差分对齐真值差分）
        if self.use_temporal:
            temp_loss = self.temporal_loss(pred, gold)
        else:
            temp_loss = torch.tensor(0.0, device=pred.device)

        # 总损失
        loss_total = (ccc_val
                      + self.mse_weight * mse_val
                      + self.smooth_weight * smooth_loss
                      + self.temporal_weight * temp_loss)

        return loss_total.float()