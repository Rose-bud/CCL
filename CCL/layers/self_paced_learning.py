# modeling/losses/self_paced_loss.py
import torch
import torch.nn as nn

class SelfPacedLoss(nn.Module):
    def __init__(self, initial_gamma=0.1, gamma_growth=1.1, max_gamma=2.0):
        super(SelfPacedLoss, self).__init__()
        self.gamma = initial_gamma
        self.gamma_growth = gamma_growth
        self.max_gamma = max_gamma
        self.epoch = 0
        
    def update_gamma(self):
        """每个epoch后更新gamma"""
        self.gamma = min(self.gamma * self.gamma_growth, self.max_gamma)
        self.epoch += 1
        
    def forward(self, losses):
        """
        Args:
            losses: tensor of shape (batch_size,) - 每个样本的损失
        Returns:
            weighted_loss: 加权后的损失
            weights: 样本权重
        """
        # 线性SPL权重函数: w_i = max(0, 1 - loss_i / gamma)
        weights = torch.clamp(1 - losses / self.gamma, min=0.0)
        
        # 加权损失 + SPL正则项
        weighted_loss = torch.mean(weights * losses) + 0.5 / self.gamma * torch.mean(weights ** 2)
        print(f"DEBUG - losses range: {losses.min():.3f} to {losses.max():.3f}")
        print(f"DEBUG - gamma: {self.gamma}")
        
        weights = torch.clamp(1 - losses / self.gamma, min=0.0)
        print(f"DEBUG - weights range: {weights.min():.3f} to {weights.max():.3f}")
        print(f"DEBUG - active samples: {(weights > 0).float().mean():.1%}")
        
        weighted_loss = torch.mean(weights * losses) + 0.5 / self.gamma * torch.mean(weights ** 2)
        print(f"DEBUG - final loss: {weighted_loss:.3f}")
        print(f"DEBUG - losses shape: {losses.shape}")  # 新增：检查形状
        print(f"DEBUG - losses dtype: {losses.dtype}")  # 新增：检查类型
        print(f"DEBUG - losses range: {losses.min():.3f} to {losses.max():.3f}")
        print(f"DEBUG - gamma: {self.gamma}")
        print(f"DEBUG - gamma type: {type(self.gamma)}")  # 新增
        
        # 检查权重计算中间结果
        ratio = losses / self.gamma
        print(f"DEBUG - loss/gamma ratio: {ratio.min():.3f} to {ratio.max():.3f}")
        
        weights = torch.clamp(1 - ratio, min=0.0)
        print(f"DEBUG - weights range: {weights.min():.3f} to {weights.max():.3f}")
    
        return weighted_loss, weights
    
    def get_gamma(self):
        return self.gamma