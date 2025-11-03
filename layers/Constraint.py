import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_constraint1(torch.nn.Module):
    '''
    该约束确保矩阵 d 的列近似正交，且每列向量的长度接近1。即使得 d 近似于正交矩阵。
    针对非稳态时序优化：添加数值稳定性
    '''
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        # 确保设备一致性
        device = d.device
        I = torch.eye(d.shape[1], device=device)
        
        # 计算正交性损失，添加数值稳定性
        d_t_d = torch.mm(d.t(), d)
        orthogonality_error = d_t_d * I - I
        
        # 限制损失上界，防止非稳态导致的爆炸
        loss_d1_constraint = torch.clamp(torch.norm(orthogonality_error), max=100.0)
        
        return 1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):
    '''
    d.shape: [20, 2]
    dim: 10
    n_clusters: 2
    该约束确保不同聚类之间的矩阵 d 的子块尽量独立，即相互正交。其目的是减少聚类之间的相互干扰。
    针对非稳态时序优化：添加数值稳定性和设备管理
    '''
    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        # 关键修复：确保所有张量在同一设备上
        device = d.device
        S = torch.ones(d.shape[1], d.shape[1], device=device)
        zero = torch.zeros(dim, dim, device=device)  # 修复：添加device参数
        
        # 将矩阵 S 按块分为 n_clusters 个大小为 dim×dim 的子矩阵，并将这些子矩阵设为 zero。
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        
        # 计算约束损失，添加数值稳定性
        d_t_d = torch.mm(d.t(), d)
        masked = d_t_d * S
        
        # 使用更稳定的范数计算，避免非稳态导致的极端值
        loss_d2_constraint = torch.clamp(torch.norm(masked), max=100.0)
        
        return 1e-3 * loss_d2_constraint


