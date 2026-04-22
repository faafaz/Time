import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances_batch(x, y=None):
    '''
    计算 Batch 级的成对距离矩阵 (完全向量化，支持 GPU)
    Input: x is (Batch, N, 1), y is (Batch, M, 1) or None
    Output: dist is (Batch, N, M)
    '''
    if y is None:
        y = x
    
    # 利用广播机制计算 (x-y)^2
    # x: (B, N, 1) -> (B, N, 1, 1)
    # y: (B, M, 1) -> (B, 1, M, 1)
    x_b = x.squeeze(-1).unsqueeze(2) # (B, N, 1)
    y_b = y.squeeze(-1).unsqueeze(1) # (B, 1, M)
    
    # 欧氏距离平方: ||x - y||^2
    dist = (x_b - y_b) ** 2
    return dist

def soft_min(x, gamma):
    # Soft-min 操作: -gamma * log(sum(exp(-x/gamma)))
    # 使用 logsumexp 保证数值稳定性
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)

class SoftDTWBatch(torch.nn.Module):
    def __init__(self, gamma=1.0):
        super(SoftDTWBatch, self).__init__()
        self.gamma = gamma

    def forward(self, D):
        """
        计算 Soft-DTW 损失 (支持 Batch 并行)
        D: (Batch, N, N) 距离矩阵
        """
        batch_size, N, M = D.shape
        device = D.device
        gamma = self.gamma

        # 初始化动态规划矩阵 R
        # R 维度为 (Batch, N+2, M+2)，加 padding 处理边界
        # 初始值为无穷大
        R = torch.full((batch_size, N + 2, M + 2), float('inf'), device=device)
        
        # 起点 (0,0) 对应的 DP 初始值为 0
        R[:, 0, 0] = 0

        # 动态规划填表
        # 虽然这里有 Python 循环，但在 Batch 维度是完全并行的
        # 对于 N=96 这样的序列长度，PyTorch 的 overhead 完全可控
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # 获取三个方向的前驱节点: (B,)
                r0 = R[:, i - 1, j - 1] # 对角
                r1 = R[:, i - 1, j]     # 上
                r2 = R[:, i, j - 1]     # 左
                
                # 拼接三个方向的值以便进行 logsumexp: (B, 3)
                concatenated = torch.stack((r0, r1, r2), dim=1)
                
                # Soft-min 更新: R[i,j] = D[i,j] + softmin(r0, r1, r2)
                R[:, i, j] = D[:, i - 1, j - 1] + soft_min(concatenated, gamma)

        # 返回终点的值
        return R[:, N, M]