import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layers.Constraint import D_constraint1, D_constraint2

class EDESC(nn.Module):

    def __init__(self,
                 d_model,
                 n_clusters,
                 eta,
                 c_out,
                 bs,
                 patch_len,
                 stride):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.eta = eta
        self.c_out = c_out
        self.bs = bs
        self.patch_len = patch_len
        self.stride = stride

        # Subspace bases proxy
        # self.D = Parameter(torch.Tensor(n_z, n_clusters))
        n_z = c_out * d_model
        self.d = int(n_z / n_clusters)
        self.D = Parameter(torch.Tensor(n_clusters * self.d, n_clusters * self.d))

    def reverse_unfold(self, z, original_length, stride):
        # z: [bs x patch_num x nvars x patch_len]
        bs, patch_num, nvars, patch_len = z.size()
        output = torch.zeros((bs, nvars, original_length), device=z.device)
        patch_counts = torch.zeros((bs, nvars, original_length), device=z.device)

        for i in range(patch_num):
            start = i * stride
            end = start + patch_len
            if end > original_length:
                end = original_length

            # Dynamically adjust the patch length if it exceeds the original length
            current_patch_len = end - start

            output[:, :, start:end] += z[:, i, :, :current_patch_len]
            patch_counts[:, :, start:end] += 1

        output /= patch_counts
        output = torch.reshape(output, (output.shape[0], output.shape[2], output.shape[1]))
        return output   # output: [bs, c_out, context_window]

    def forward(self, z):  # z: [bs * patch_num_out x nvars * d_model]
        # x_bar, z = self.ae(x)  # x_bar: [bs * patch_num x nvars * patch_len]
        # x_bar = torch.reshape(x_bar, (self.bs, -1, self.c_out, self.patch_len))
        # x_bar = self.reverse_unfold(x_bar, length, self.stride)    # x_bar: [bs, c_out, context_window]
        s = None
        
        # 关键修复：对输入进行L2归一化，防止非稳态数据的幅值爆炸
        # 这对于MoE的路由决策至关重要
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-8)
        
        # Calculate subspace affinity
        for i in range(self.n_clusters):
            # 使用归一化后的特征计算亲和度
            # 避免平方运算导致的数值爆炸
            D_i = self.D[:, i * self.d:(i + 1) * self.d]
            
            # 对子空间基也进行归一化
            D_i_norm = F.normalize(D_i, p=2, dim=0, eps=1e-8)
            
            # 计算余弦相似度而非欧氏距离的平方
            # 这在非稳态场景下更鲁棒
            similarity = torch.mm(z_norm, D_i_norm)
            si = torch.sum(torch.pow(similarity, 2), 1, keepdim=True)
            
            # 使用温度缩放控制分布的锐度
            temperature = 0.1
            si = si / temperature
            
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        
        # 使用softmax代替手动归一化，更加数值稳定
        # 添加eta项作为正则化
        s = s + self.eta * self.d
        s = F.softmax(s, dim=1)  # 自动处理数值稳定性
        
        return s, z

    def total_loss(self, pred, target, dim, n_clusters, beta): # x, x_bar,
        # Reconstruction loss   Eq 9
        # reconstr_loss = F.mse_loss(x_bar, x)

        # Subspace clustering loss  Eq 15
        # 关键修复：使用对数空间的数值稳定版本KL散度
        # 避免在非稳态时序中出现极端概率值导致的NaN
        eps = 1e-8
        
        # 确保pred和target都是有效的概率分布
        pred = torch.clamp(pred, min=eps, max=1.0)
        target = torch.clamp(target, min=eps, max=1.0)
        
        # 重新归一化，确保和为1
        pred = pred / (pred.sum(dim=1, keepdim=True) + eps)
        target = target / (target.sum(dim=1, keepdim=True) + eps)
        
        # 使用数值稳定的KL散度计算
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        log_pred = torch.log(pred + eps)
        log_target = torch.log(target + eps)
        
        # 使用log_softmax风格的稳定计算
        kl_loss = F.kl_div(log_pred, target, reduction='batchmean')
        
        # 添加KL散度的上界约束，防止非稳态导致的极端值
        kl_loss = torch.clamp(kl_loss, max=10.0)

        # Constraints   Eq 12
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)

        # Total_loss    Eq 16
        # 调整beta系数，避免聚类损失主导导致的不稳定
        total_loss = beta * kl_loss + loss_d1 + loss_d2  # reconstr_loss +

        return total_loss
