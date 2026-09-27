"""
适配RGB特征的Visual-Linguistic Verification模块
基于VLTVG论文实现，适配您的输入输出维度要求

输入:
- RGB_cash: [B, 512, 128]  # 视觉特征，feat_dim=512, patches=128
- RGB_t_feas: [B, 77, 512]  # 文本特征，77个token，每个512维

输出:
- vlv_rgb_cls: [B, 1, 512]  # CLS特征
- vlv_rgb_S: [B, 128]       # 验证分数 (每个patch一个分数)
- rgb_all: [B, 128, 512]    # 调制后的视觉特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 不在最后一层添加激活函数
                layers.append(QuickGELU())
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [B, N, C] 视觉特征
            key: [B, L, C] 文本特征
            value: [B, L, C] 文本特征
            key_padding_mask: [B, L] 文本掩码
        """
        B, N, C = query.shape
        _, L, _ = key.shape
        
        # 线性投影
        q = self.q_proj(query).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, N, head_dim]
        k = self.k_proj(key).view(B, L, self.nhead, self.head_dim).transpose(1, 2)    # [B, nhead, L, head_dim]
        v = self.v_proj(value).view(B, L, self.nhead, self.head_dim).transpose(1, 2)   # [B, nhead, L, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nhead, N, L]
        
        # 应用掩码
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, v)  # [B, nhead, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        
        # 输出投影
        out = self.out_proj(out)
        
        return out, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制（带相对位置编码）"""
    def __init__(self, d_model=512, nhead=8, dropout=0.1, rel_pos_bins=32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rel_pos_bins = rel_pos_bins
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置偏置
        if rel_pos_bins > 0:
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.nhead, rel_pos_bins))
            nn.init.normal_(self.rel_pos_bias, std=0.01)
        
    def _get_rel_pos_idx(self, N):
        """计算相对位置索引"""
        idx = torch.arange(N).unsqueeze(0) - torch.arange(N).unsqueeze(1)  # [N, N]
        idx = idx.abs().clamp(max=self.rel_pos_bins - 1).long()
        return idx
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, N, C] 输入特征
            key_padding_mask: [B, N] 掩码
        """
        B, N, C = x.shape
        
        # 线性投影
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, N, head_dim]
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, N, head_dim]
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, N, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nhead, N, N]
        
        # 添加相对位置偏置
        if self.rel_pos_bins > 0:
            rel_pos_idx = self._get_rel_pos_idx(N).to(x.device)  # [N, N]
            rel_pos_bias = self.rel_pos_bias[:, rel_pos_idx]  # [nhead, N, N]
            attn_scores = attn_scores + rel_pos_bias.unsqueeze(0)  # [B, nhead, N, N]
        
        # 应用掩码
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, v)  # [B, nhead, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        
        # 输出投影
        out = self.out_proj(out)
        
        return out, attn_weights


class RGBVisualLinguisticVerification(nn.Module):
    """
    适配RGB特征的Visual-Linguistic Verification模块
    
    输入:
        RGB_cash: [B, 512, 128]  # 视觉特征
        RGB_t_feas: [B, 77, 512] # 文本特征
    
    输出:
        vlv_rgb_cls: [B, 1, 512]  # CLS特征
        vlv_rgb_S: [B, 128]       # 验证分数
        rgb_all: [B, 128, 512]    # 调制后的视觉特征
    """
    
    def __init__(self, 
                 feat_dim=512,
                 text_dim=512,
                 nhead=8,
                 dropout=0.1,
                 rel_pos_bins=32,
                 init_alpha=1.0,
                 init_sigma=0.5,cfg=None):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.warmup = int(cfg.SOLVER.SPL_WARMUP) if cfg is not None else 5
        
        # Visual-linguistic verification module - 第一个cross-attention
        self.img2text_attn = MultiHeadCrossAttention(feat_dim, nhead, dropout)
        # Language-guided context encoder - 第二个cross-attention
        self.img2textcond_attn = MultiHeadCrossAttention(feat_dim, nhead, dropout)
        self.context_self_attn = MultiHeadSelfAttention(feat_dim, nhead, dropout, rel_pos_bins)
        
        # 投影层用于验证分数计算
        self.visual_proj = MLP(feat_dim, feat_dim, feat_dim, num_layers=1)
        self.semantic_proj = MLP(feat_dim, feat_dim, feat_dim, num_layers=1)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)
         # 自适应缩放层
        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        self.ff = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim*2),
                        QuickGELU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.feat_dim*2, self.feat_dim))
        # CLS特征投影
        self.cls_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, RGB_cash, RGB_t_feas, text_mask=None,use_cls_token=True,sample_weights=None,epoch=0,ppp=4):
         # 自动检测 RGB_cash 的格式并转换
        B, dim1, dim2 = RGB_cash.shape
        
        # 判断格式：[B, feat_dim, patches] 还是 [B, patches, feat_dim]
        if dim1 == self.feat_dim and dim2 != self.feat_dim:
            # [B, feat_dim, patches] 格式，需要转换
            patches = dim2
            feat_dim = dim1
            visual_feat = RGB_cash.permute(0, 2, 1).contiguous()  # [B, patches, feat_dim]
        elif dim2 == self.feat_dim and dim1 != self.feat_dim:
            # [B, patches, feat_dim] 格式，直接使用
            patches = dim1
            feat_dim = dim2
            visual_feat = RGB_cash  # [B, patches, feat_dim]
        else:
            raise ValueError(f"无法确定 RGB_cash 的格式: {RGB_cash.shape}, feat_dim={self.feat_dim}")
        
        v_tokens = visual_feat
        # --- ensure mask boolean ---
        if text_mask is not None:
            text_mask = text_mask.to(torch.bool)

        # 1) 第一个 cross-attn: visual <- text
        att_out, _ = self.img2text_attn(query=visual_feat, key=RGB_t_feas, value=RGB_t_feas, key_padding_mask=text_mask)
        # residual + norm
        visual_feat = visual_feat + att_out
        visual_feat = self.norm1(visual_feat)

        # 计算验证分数（先投影）
        visual_proj = self.visual_proj(visual_feat)      # [B, 128, 512]
        # 使用 text-guided features: 先用另一个 cross-attn 获得 text_info_for_score
        text_info_for_score, _ = self.img2text_attn(query=visual_feat, key=RGB_t_feas, value=RGB_t_feas, key_padding_mask=text_mask)
        semantic_proj = self.semantic_proj(text_info_for_score)  # [B,128,512]

        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        semantic_norm = F.normalize(semantic_proj, p=2, dim=-1)
        cosine_sim = (visual_norm * semantic_norm).sum(dim=-1, keepdim=True)  # [B,128,1]
        verify_score = self.alpha * torch.exp(-(1 - cosine_sim).pow(2) / (2 * self.sigma**2))  # [B,128,1]
        # 2) language-guided context encoder
        text_cond_info, _ = self.img2textcond_attn(query=visual_feat, key=RGB_t_feas, value=RGB_t_feas, key_padding_mask=text_mask)
        q = visual_feat + text_cond_info
        context_feat, self_attn = self.context_self_attn(q)  # [B,128,512]

        # residual + norm + small FFN
        context_feat = q + context_feat
        context_feat = self.norm2(context_feat)

        context_feat = context_feat + self.ff(context_feat)

        # modulation
        modulated_feat = (self.norm1(visual_feat) + self.norm2(context_feat)) * verify_score  # broadcast
        modulated_feat = self.norm3(modulated_feat)
        
        # cls pooling by verify -> 注意 verify_score_flat 非负
        # final modulation 
        verify_score_flat = verify_score.squeeze(-1)  # [B,128]
        attn_weights = F.softmax(verify_score_flat, dim=-1).unsqueeze(-1)  # [B,128,1]
        cls_feat = (modulated_feat * attn_weights).sum(dim=1)  # [B,512]
        cls_feat = self.cls_proj(cls_feat).unsqueeze(1)  # [B,1,512]
        gate = 1
        w_expanded=1
        if sample_weights is not None:
            gate =1
            w_expanded = sample_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        else:
            gate=0.0
        if epoch <= self.warmup:
            gate=0.2
            w_expanded=1
        modulated_feat = gate *w_expanded* modulated_feat + (1 - gate*w_expanded) * v_tokens
        additional_info = {
            "attn_cross": None,  # 可以添加交叉注意力权重
            "attn_self": None,   # 可以添加自注意力权重
            "verification_loss": 0.0,  # 验证损失（如果需要）
            "dot": None,         # dot product（如果需要）
            "S": verify_score.squeeze(-1),  # 验证分数
        }
        return cls_feat, verify_score_flat, modulated_feat, additional_info
        
    


# 使用示例和测试
if __name__ == "__main__":
    # 创建模型
    model = RGBVisualLinguisticVerification(
        feat_dim=512,
        text_dim=512,
        nhead=8,
        dropout=0.1
    )
    
    # 模拟输入数据
    batch_size = 2
    RGB_cash = torch.randn(batch_size, 512, 128)    # [B, 512, 128]
    RGB_t_feas = torch.randn(batch_size, 77, 512)    # [B, 77, 512]
    
    print(f"输入视觉特征形状: {RGB_cash.shape}")
    print(f"输入文本特征形状: {RGB_t_feas.shape}")
    
    # 前向传播
    with torch.no_grad():
        vlv_rgb_cls, vlv_rgb_S, rgb_all, _ = model(RGB_cash, RGB_t_feas)
    
    print(f"输出CLS特征形状: {vlv_rgb_cls.shape}")
    print(f"输出验证分数形状: {vlv_rgb_S.shape}")
    print(f"输出调制特征形状: {rgb_all.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 验证输出维度
    assert vlv_rgb_cls.shape == (batch_size, 1, 512), f"CLS特征形状错误: {vlv_rgb_cls.shape}"
    assert vlv_rgb_S.shape == (batch_size, 128), f"验证分数形状错误: {vlv_rgb_S.shape}"
    assert rgb_all.shape == (batch_size, 128, 512), f"调制特征形状错误: {rgb_all.shape}"
    
    print("✅ 所有输出维度验证通过!")
    print(f"验证分数范围: [{vlv_rgb_S.min():.4f}, {vlv_rgb_S.max():.4f}]")
    print(f"验证分数均值: {vlv_rgb_S.mean():.4f}")
