import torch
import torch.nn.functional as F

def vmd_loss(x, modes, lambda_ortho=0.1, lambda_sparse=0.01):
    # 1. 重构损失
    l_recon = F.mse_loss(modes.sum(dim=-1), x.squeeze(-1))
    
    # 2. 正交损失 (防止模态混叠)
    l_ortho = orthogonality_loss(modes)
    
    # 3. 稀疏损失 (让多余的 K 自动归零)
    # l_sparse = sparsity_loss(modes)
    entropy = vmd_energy_balancing_loss(modes)
    # print('recon loss: ', l_recon, 'ortho loss: ', l_ortho, 'entropy loss: ',  entropy)
   
    # 4. 汇总
    total_loss = l_recon + lambda_ortho * l_ortho + entropy * 0.1
    
    return total_loss

def orthogonality_loss(modes):
    # modes: [B, L, K] -> 转为频域进行正交性计算更准确
    modes_f = torch.fft.rfft(modes, dim=1)
    modes_f_abs = torch.abs(modes_f) # [B, Nf, K]
    
    # 计算 K 个模态之间的相关矩阵
    # [B, K, Nf] @ [B, Nf, K] -> [B, K, K]
    correlation_matrix = torch.bmm(modes_f_abs.transpose(1, 2), modes_f_abs)
    
    # 我们希望对角线以外的元素（不同模态间的重叠）越小越好
    eye = torch.eye(modes.shape[-1], device=modes.device).unsqueeze(0)
    off_diagonal = correlation_matrix * (1 - eye)
    return off_diagonal.mean()

def sparsity_loss(modes):
    # 计算每个模态的 L2 能量
    energies = torch.norm(modes, p=2, dim=1) # [B, K]
    # 对能量向量求 L1 范数，促使部分模态能量趋近于 0
    return torch.mean(torch.sum(torch.abs(energies), dim=1))

def vmd_energy_balancing_loss(modes):
    # modes: [B, L, K]
    # 1. 计算每个模态的能量
    energies = torch.norm(modes, p=2, dim=1) # [B, K]
    # 2. 计算能量分布的概率密度
    p = energies / (energies.sum(dim=1, keepdim=True) + 1e-8)
    # 3. 计算熵：熵越大，分布越均匀
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
    # 我们希望熵大，所以 Loss 是负熵
    return -entropy