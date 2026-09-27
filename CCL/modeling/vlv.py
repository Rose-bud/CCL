# models/vlv_vltvg_for_demo.py
"""
VLV module re-implemented with VLTVG paper details:
- Visual-Linguistic Verification (per-patch attention + verification score)
- Language-Guided Context Encoder:
    1) cross-attn: Fs (text -> per-patch semantic)
    2) self-attn on (Fv + Fc) with optional relative positional bias W_KR(i-j)
- Designed to accept DeMo / CLIP ViT tokens:
    - visual_tokens: [B, C, N] or [B, N, C]
    - text_feats: [B, L, Tdim] (token-level) or [B, Tdim] pooled
- Returns:
    - vlv_cls: [B,1,C]
    - S_map: [B, N]  (per-patch verification score)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(module):
    # helper for stable init
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    if isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        hidden = int(dim * mlp_ratio)
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.apply(xavier_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttentionWithRelPos(nn.Module):
    """
    Self-attention with optional relative-position bias W_KR(i-j).
    If rel_pos_bins is None -> no relative bias used.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, rel_pos_bins=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # relative position bias table (learnable)
        self.rel_pos_bins = rel_pos_bins
        if rel_pos_bins is not None and rel_pos_bins > 0:
            # we implement simple 1D relative bins: distance -> index
            self.rel_bias = nn.Parameter(torch.zeros(self.num_heads, rel_pos_bins))
            nn.init.normal_(self.rel_bias, std=1e-6)

        self.apply(xavier_init)

    def forward(self, x, key_padding_mask=None, rel_pos_idx=None):
        # x: [B, N, C]
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, N, Dh
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # B, H, N, N

        if self.rel_pos_bins is not None and rel_pos_idx is not None:
            # rel_pos_idx: [N, N] indices into range [0, rel_pos_bins)
            # add bias per head
            # rel_bias shape: [H, rel_pos_bins] -> indexed -> [H, N, N]
            bias = self.rel_bias[:, rel_pos_idx]  # [H, N, N]
            attn = attn + bias.unsqueeze(0)

        if key_padding_mask is not None:
            # key_padding_mask: [B, N] True for padding -> mask out
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # B, H, N, Dh
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out(out)
        out = self.dropout(out)
        return out, attn


class CrossAttentionKV(nn.Module):
    """
    Cross-attention: queries from visual tokens, keys/values from text.
    Uses standard MHA (via linear q,k,v).
    """
    def __init__(self, dim, text_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        # linear projections: we will project text_dim -> dim
        if text_dim != dim:
            self.text_proj = nn.Linear(text_dim, dim)
        else:
            self.text_proj = nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.apply(xavier_init)

    def forward(self, q_src, kv_src, key_padding_mask=None):
        # q_src: [B, Nq, C] visual queries
        # kv_src: [B, L, Tdim]
        B, Nq, C = q_src.shape
        kv = self.text_proj(kv_src)  # [B, L, C]
        # q,k,v
        q = self.q(q_src).view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(kv).view(B, kv.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(kv).view(B, kv.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # B,H,Nq,L
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # B,H,Nq, Dh
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
        out = self.out(out)
        out = self.dropout(out)
        return out, attn


class LanguageGuidedContextEncoder(nn.Module):
    """
    Implements VLTVG's language-guided context encoder:
      - Fs = cross-attn(Fv, Fl)
      - Fc: then do self-attn on (Fv + Fs) with relative pos bias
    """
    def __init__(self, feat_dim, text_dim, num_heads=8, rel_pos_bins=None, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttentionKV(dim=feat_dim, text_dim=text_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn = MultiHeadSelfAttentionWithRelPos(dim=feat_dim, num_heads=num_heads, dropout=dropout,
                                                          rel_pos_bins=rel_pos_bins)
        self.ln1 = nn.LayerNorm(feat_dim)
        self.ln2 = nn.LayerNorm(feat_dim)
        self.ffn = MLP(feat_dim, mlp_ratio=4.0, dropout=dropout)
        self.apply(xavier_init)

    def forward(self, Fv, Fl, text_mask=None, rel_pos_idx=None):
        # Fv: [B, N, C], Fl: [B, L, Tdim]
        Fs, attn1 = self.cross_attn(Fv, Fl, key_padding_mask=text_mask)  # B,N,C
        # language-guided self-attn: query/key = (Fv + Fs)
        qk = Fv + Fs
        qk = self.ln1(qk)
        Fc, attn2 = self.self_attn(qk, key_padding_mask=None, rel_pos_idx=rel_pos_idx)
        # add residual and FFN
        x = qk + Fc
        x = x + self.ffn(self.ln2(x))
        return x, attn1, attn2  # x is language-guided context features (B,N,C)


class VisualLinguisticVerificationVLTVG(nn.Module):
    """
    Combined V-L Verification + Language-guided Context, adapted to DeMo/CLIP outputs.
    """
    def __init__(self,
                 feat_dim: int,
                 text_dim: int = None,
                 nhead: int = 8,
                 pooling: str = "avg",
                 rel_pos_bins: int = 32,
                 learnable_alpha: bool = True,
                 init_alpha: float = 1.0,
                 learnable_log_sigma: bool = True,
                 init_sigma: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        if text_dim is None:
            text_dim = feat_dim
        assert pooling in ("avg", "cls", "attn")
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.pooling = pooling

        # textual projection happens inside CrossAttentionKV
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        # verification projection (L2 norm after)
        self.v_ver_proj = nn.Linear(feat_dim, feat_dim)
        self.s_ver_proj = nn.Linear(feat_dim, feat_dim)

        # language-guided context encoder (VLTVG)
        self.context_encoder = LanguageGuidedContextEncoder(feat_dim, text_dim, num_heads=nhead,
                                                            rel_pos_bins=rel_pos_bins, dropout=dropout)

        # pooling projection
        self.pool_proj = nn.Linear(feat_dim, feat_dim)

        # learnable alpha & log sigma for verification score
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(init_alpha, dtype=torch.float32))
        if learnable_log_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma, dtype=torch.float32)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(init_sigma, dtype=torch.float32)))

        self.dropout = nn.Dropout(dropout)
        self.apply(xavier_init)

    def _to_tokens_first(self, visual_tokens):
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be 3D")
        B, a, b = visual_tokens.shape
        if a == self.feat_dim:
            # [B, C, N] -> [B, N, C]
            return visual_tokens.permute(0, 2, 1).contiguous()
        if b == self.feat_dim:
            return visual_tokens
        raise ValueError("visual_tokens shape ambiguous")

    def _make_rel_pos_idx(self, N, bins):
        """
        Build simple relative position index [N,N] mapping distances to bins (clipped).
        bins: number of bins
        """
        # use distance matrix (absolute difference) and clip to bins-1
        idx = torch.arange(N).unsqueeze(0) - torch.arange(N).unsqueeze(1)  # N,N (i-j)
        idx = idx.abs().clamp(max=bins - 1).long()
        return idx  # cpu tensor; will be handled in forward (broadcasted to device)

    def forward(self, visual_tokens, text_feats, text_mask=None, use_cls_token: bool = False, cls_token: torch.Tensor = None):
        """
        visual_tokens: [B, C, N] or [B, N, C]
        text_feats: [B, L, Tdim] or [B, Tdim]
        returns: vlv_cls [B,1,C], S_map [B,N], v_mod [B,N,C], and optionally intermediate maps via debug flags
        """
        v_tokens = self._to_tokens_first(visual_tokens)  # [B,N,C]
        B, N, C = v_tokens.shape
        # compute Fs via cross-attn and language-guided context features Fc
        # ensure text_feats passed are [B,L,Tdim] or [B,Tdim]
        if text_feats is None:
            raise ValueError("text_feats required.")
        # compute relative idx if needed
        rel_pos_idx = None
        # if context encoder uses rel bins, prepare idx
        if isinstance(self.context_encoder.self_attn.rel_pos_bins, int) and self.context_encoder.self_attn.rel_pos_bins > 0:
            rel_pos_idx = self._make_rel_pos_idx(N, self.context_encoder.self_attn.rel_pos_bins).to(v_tokens.device)

        Fc, attn_cross, attn_self = self.context_encoder(v_tokens, text_feats, text_mask=text_mask, rel_pos_idx=rel_pos_idx)  # [B,N,C]

        # Visual-linguistic verification: compute semantic map Fs from cross-attn outputs
        # Here we reuse cross-attn output (attn_cross) -> for strictness also project
        # Recompute Fs by using CrossAttentionKV to gather textual semantics per-patch (we already did inside context_encoder.cross_attn,
        # we can either re-use attn or recompute via dedicated cross-attn; reuse to save computation)
        Fs = Fc  # using Fc (language-guided context) as richer semantic map for verification

        # Projection + L2 normalization prior to dot product as paper
        v_proj = F.normalize(self.v_ver_proj(v_tokens), dim=-1)  # [B,N,C]
        s_proj = F.normalize(self.s_ver_proj(Fs), dim=-1)       # [B,N,C]

        dot = (v_proj * s_proj).sum(dim=-1)   # [B,N], in [-1,1]
        sigma = torch.exp(self.log_sigma)
        # 修复验证分数计算：使用更合理的公式
        # 当 dot 接近 1 时，S 接近 alpha；当 dot 接近 -1 时，S 接近 0
        S = self.alpha * torch.sigmoid((dot - 0.5) / sigma)  # [B,N]
        print(f"VLV Debug - dot: {dot.mean():.4f}±{dot.std():.4f}, sigma: {sigma:.4f}, alpha: {self.alpha:.4f}, S: {S.mean():.4f}±{S.std():.4f}")
        print(f"  dot range: [{dot.min():.4f}, {dot.max():.4f}], S range: [{S.min():.4f}, {S.max():.4f}]")
        # modulate visual tokens by verification score and also add context features
        v_mod = (v_tokens + Fc) * S.unsqueeze(-1)  # [B,N,C]

        # pooling to get cls-like token
        if self.pooling == "avg":
            pooled = v_mod.mean(dim=1)  # [B,C]
        elif self.pooling == "cls" and use_cls_token and cls_token is not None:
            pooled_mod = v_mod.mean(dim=1)
            pooled = cls_token.squeeze(1) + pooled_mod
        elif self.pooling == "attn":
            # attention pooling: compute scores based on verification scores S
            # Use verification scores S as attention weights (they already indicate patch importance)
            att = F.softmax(S, dim=-1).unsqueeze(-1)  # [B,N,1]
            pooled = (v_mod * att).sum(dim=1)  # [B,C]
        else:
            pooled = v_mod.mean(dim=1)
        # 添加基于平均验证分数的残差连接
        w = S.mean(dim=1, keepdim=True)  # [B, 1] - 每个样本的平均验证分数
        w = w.unsqueeze(-1)  # [B, 1, 1] - 扩展到与v_mod和v_tokens匹配的维度
        v_mod = v_mod + (1 - w) * v_tokens  # 加权残差连接
        vlv_cls = self.pool_proj(self.dropout(pooled)).unsqueeze(1)  # [B,1,C]
        # return verification map and optionally attention maps for visualization
        return vlv_cls, S, v_mod, {"attn_cross": attn_cross, "attn_self": attn_self}
