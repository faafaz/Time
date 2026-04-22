# ... existing code ...
import numpy as np
import torch
from numba import jit
from torch.autograd import Function

def pairwise_distances(x, y=None):
    '''
    计算两个矩阵的成对欧氏距离平方（高效向量化实现）
    
    参数:
        x: Nxd 矩阵 (N个样本, d维特征)
        y: Mxd 矩阵 (可选, 默认为x)
    
    返回:
        NxM 距离矩阵 dist, 其中 dist[i,j] = ||x[i,:] - y[j,:]||^2
    
    原理:
        利用 (a-b)^2 = a^2 - 2ab + b^2 展开式
        x_norm = Σx_i^2 (每行平方和)
        y_norm = Σy_j^2 (每行平方和)
        通过矩阵乘法计算内积项: x·y^T
    '''
    x_norm = (x**2).sum(1).view(-1, 1)  # 计算x每行的平方和 [N, 1]
    if y is not None:
        y_t = torch.transpose(y, 0, 1)   # 转置y为dxM
        y_norm = (y**2).sum(1).view(1, -1)  # 计算y每行的平方和 [1, M]
    else:
        y_t = torch.transpose(x, 0, 1)   # y=x时的特殊情况
        y_norm = x_norm.view(1, -1)      # 重用x的范数
    
    # 核心计算: ||x-y||^2 = x^2 - 2xy^T + y^2
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))  # 确保非负（数值稳定性）

@jit(nopython=True)
def compute_softdtw(D, gamma):
    """
    Soft-DTW前向计算（动态规划）
    
    参数:
        D: NxM 距离矩阵 (两个时间序列的成对距离)
        gamma: 平滑参数 (gamma→0时退化为标准DTW)
    
    返回:
        R: (N+2)x(M+2) 动态规划矩阵 (含边界填充)
    
    算法原理:
        1. R[i,j] = 从(0,0)到(i-1,j-1)的最小累积距离
        2. 使用softmin替代min实现可微分:
           softmin(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
        3. 边界条件: R[0,0]=0, 其他边界设为极大值(1e8)
    """
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8  # 初始化动态规划矩阵(含边界)
    R[0, 0] = 0  # 起点距离为0
    
    # 填充动态规划矩阵
    for j in range(1, M + 1):          # 遍历目标序列
        for i in range(1, N + 1):      # 遍历源序列
            # 获取三个可能路径的累积距离
            r0 = -R[i - 1, j - 1] / gamma  # 对角线(匹配当前点)
            r1 = -R[i - 1, j] / gamma      # 向下(跳过源序列点)
            r2 = -R[i, j - 1] / gamma      # 向右(跳过目标序列点)
            
            # 数值稳定化处理: 减去最大值防止溢出
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            
            # 计算softmin(平滑的最小值)
            softmin = -gamma * (np.log(rsum) + rmax)
            
            # 更新累积距离 = 当前距离 + softmin(历史路径)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R

@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma):
    """
    Soft-DTW反向传播梯度计算
    
    参数:
        D_: NxM 原始距离矩阵
        R: (N+2)x(M+2) 前向计算的动态规划矩阵
        gamma: 平滑参数
    
    返回:
        E: NxM 梯度矩阵 (指示每个距离对最终损失的贡献)
    
    反向传播原理:
        1. 从终点反向遍历动态规划路径
        2. 计算每个位置的梯度贡献:
           E[i,j] = Σ(路径概率 * 后续梯度)
        3. 路径概率通过softmax计算:
           P(对角线) = exp(-a/gamma)/Z, 其中Z为归一化因子
    """
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_  # 填充原始距离矩阵(含边界)
    E[-1, -1] = 1  # 终点梯度初始化为1
    
    # 设置无效区域为负无穷(防止反向传播到边界)
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]  # 保持终点值
    
    # 反向动态规划
    for j in range(M, 0, -1):      # 从终点反向遍历
        for i in range(N, 0, -1):
            # 计算三条路径的"能量" (未归一化概率)
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            
            # 计算softmax概率
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            
            # 累积梯度 = 各路径概率 * 后续梯度
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    
    return E[1:N + 1, 1:M + 1]  # 返回有效区域梯度

class SoftDTWBatch(Function):
    """批量Soft-DTW损失函数 (支持GPU加速)"""
    
    @staticmethod
    def forward(ctx, D, gamma=1.0):
        """
        前向传播: 计算批量Soft-DTW距离
        
        参数:
            D: [batch_size, N, N] 批量距离矩阵
            gamma: 平滑参数
        
        返回:
            批量平均Soft-DTW距离
        """
        dev = D.device
        batch_size, N, N = D.shape
        gamma_tensor = torch.FloatTensor([gamma]).to(dev)
        D_np = D.detach().cpu().numpy()  # 转换为numpy进行JIT加速计算
        
        total_loss = 0
        R = torch.zeros((batch_size, N+2, N+2)).to(dev)  # 存储所有样本的R矩阵
        
        # 逐样本计算 (numba不支持批量处理)
        for k in range(batch_size):
            # 计算单个样本的动态规划矩阵
            Rk = torch.FloatTensor(compute_softdtw(D_np[k, :, :], gamma)).to(dev)
            R[k:k+1, :, :] = Rk  # 保存中间结果
            total_loss += Rk[-2, -2]  # 累加最终距离
        
        # 保存反向传播所需上下文
        ctx.save_for_backward(D, R, gamma_tensor)
        return total_loss / batch_size  # 返回平均损失
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播: 计算距离矩阵的梯度
        
        参数:
            grad_output: 损失对输出的梯度 (标量)
        
        返回:
            D的梯度 (与输入D形状相同)
        """
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, N = D.shape
        D_np = D.detach().cpu().numpy()
        R_np = R.detach().cpu().numpy()
        gamma_val = gamma.item()
        
        # 初始化梯度矩阵
        E = torch.zeros((batch_size, N, N)).to(dev)
        
        # 逐样本计算梯度
        for k in range(batch_size):
            # 计算单个样本的梯度
            Ek = torch.FloatTensor(
                compute_softdtw_backward(D_np[k, :, :], R_np[k, :, :], gamma_val)
            ).to(dev)
            E[k:k+1, :, :] = Ek
        
        # 链式法则: dL/dD = grad_output * d(SoftDTW)/dD
        return grad_output * E, None
# ... existing code ...