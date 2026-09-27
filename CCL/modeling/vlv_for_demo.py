# models/vlv_for_demo.py
"""
Visual-Linguistic Verification module adapted for DeMo/CLIP outputs.

Usage summary (integration):
- visual_tokens: RGB_cash as returned by BACKBONE.
    NOTE: In DeMo's code RGB_cash often has shape [B, C, N] (channels-first).
    This module accepts either:
        - [B, C, N]  (channels-first)
        - [B, N, C]  (tokens-first)
- text_feats: RGB_t_feas as returned by BACKBONE.
    This can be either:
        - token-level text embeddings: [B, L, Tdim]
        - pooled text embedding: [B, Tdim]
- Returns:
    - vlv_cls: [B, 1, C]  (cls-like text-aware visual feature — ready to be used as an "expert token")
    - S_map: [B, N]       (per-patch verification scores — useful for visualization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualLinguisticVerificationForDeMo(nn.Module):
    def __init__(self,
                 feat_dim: int,         # C, should match DeMo self.feat_dim
                 text_dim: int = None,  # if None, will assume same as feat_dim
                 nhead: int = 8,
                 pooling: str = "avg",  # 'avg' | 'cls' | 'attn'
                 learnable_alpha: bool = True,
                 init_alpha: float = 1.0,
                 learnable_log_sigma: bool = True,
                 init_sigma: float = 0.1,
                 use_context_mlp: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert pooling in ("avg", "cls", "attn")
        self.feat_dim = feat_dim
        self.pooling = pooling
        self.nhead = nhead

        if text_dim is None:
            text_dim = feat_dim
        self.text_dim = text_dim

        # projection from text_dim -> feat_dim (if needed)
        if text_dim != feat_dim:
            self.text_proj = nn.Linear(text_dim, feat_dim)
        else:
            self.text_proj = nn.Identity()

        # project visual tokens (optional, keep dims consistent)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

        # MultiheadAttention (batch_first=True)
        # We'll use PATCH tokens as queries (compute text-aware semantic map)
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=nhead, batch_first=True, dropout=dropout)

        # optional small context MLP applied to semantic map
        if use_context_mlp:
            self.context_mlp = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim)
            )
        else:
            self.context_mlp = nn.Identity()

        # pooling projection to produce cls-like output
        self.pool_proj = nn.Linear(feat_dim, feat_dim)

        # alpha and sigma parameters for verification score
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(init_alpha, dtype=torch.float32))

        if learnable_log_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma, dtype=torch.float32)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(init_sigma, dtype=torch.float32)))

    def _normalize_to_tokens_first(self, visual_tokens):
        """
        Accept either [B, C, N] or [B, N, C], return [B, N, C].
        """
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be 3D tensor")
        B, a, b = visual_tokens.shape
        # heuristic: if middle dim equals feat_dim -> channels-first [B, C, N]
        if a == self.feat_dim:
            # assume [B, C, N] -> convert
            return visual_tokens.permute(0, 2, 1).contiguous()  # [B, N, C]
        # else assume [B, N, C]
        if b == self.feat_dim:
            return visual_tokens
        # if ambiguous, try to adapt by reshaping? but safer to error
        raise ValueError(f"visual_tokens shape ambiguous or incompatible: {visual_tokens.shape}, expected feat_dim={self.feat_dim}")

    def forward(self, visual_tokens, text_feats, text_mask=None, use_cls_token: bool = False, cls_token: torch.Tensor = None):
        """
        visual_tokens: [B, C, N] or [B, N, C]
        text_feats:    [B, L, Tdim] (token-level)  OR  [B, Tdim] (pooled)
        text_mask:     [B, L] boolean mask (True for padding) ; compatible with nn.MultiheadAttention key_padding_mask
        use_cls_token: if True and cls_token provided, cls_token should be [B,1,C] and will be used in 'cls' pooling mode
        cls_token: optional class token to use when pooling='cls' (e.g., if your backbone returns cls separately)
        Returns:
            vlv_cls: [B,1,C]
            S_map: [B, N]   (per-patch score)
        """
        # normalize input visual tokens to [B, N, C]
        v_tokens = self._normalize_to_tokens_first(visual_tokens)  # [B,N,C]
        B, N, C = v_tokens.shape
        assert C == self.feat_dim, f"expected feat_dim {self.feat_dim}, got {C}"

        # prepare text keys/values
        # Accept either [B, L, Tdim] or [B, Tdim]
        if text_feats is None:
            raise ValueError("text_feats must be provided (from BACKBONE's RGB_t_feas)")

        if text_feats.ndim == 2:
            # pooled text vector [B, Tdim] -> create single key/value
            kv = self.text_proj(text_feats).unsqueeze(1)  # [B,1,C]
            kv_mask = None
        elif text_feats.ndim == 3:
            kv = self.text_proj(text_feats)  # [B, L, C]
            kv_mask = text_mask
        else:
            raise ValueError("text_feats must be 2D or 3D tensor")

        # Query: patch tokens (projected)
        q = self.v_proj(v_tokens)  # [B,N,C]

        # MultiHeadAttention: query=q, key=kv, value=kv
        # key_padding_mask uses True for positions that should be ignored
        attn_out, attn_weights = self.mha(query=q, key=kv, value=kv, key_padding_mask=text_mask)  # attn_out: [B,N,C]

        # Fs semantic map
        Fs = attn_out  # [B, N, C]

        # compute per-patch verification score S
        v_norm = F.normalize(q, dim=-1)     # [B,N,C]
        s_norm = F.normalize(Fs, dim=-1)    # [B,N,C]
        dot = (v_norm * s_norm).sum(dim=-1)  # [B,N]
        sigma = torch.exp(self.log_sigma)
        S = self.alpha * torch.exp(- ((1.0 - dot) ** 2) / (2.0 * (sigma ** 2) + 1e-12))  # [B,N]

        # context vector and modulation
        vc = self.context_mlp(Fs)  # [B,N,C]
        v_mod = (v_tokens + vc) * S.unsqueeze(-1)  # [B,N,C]

        # pooling into a single vector per image
        if self.pooling == "avg":
            pooled = v_mod.mean(dim=1)  # [B,C]
        elif self.pooling == "cls":
            if use_cls_token and (cls_token is not None):
                # cls_token expected [B,1,C] -> squeeze
                pooled_mod = v_mod.mean(dim=1)
                pooled = cls_token.squeeze(1) + pooled_mod  # [B,C]
            else:
                pooled = v_mod.mean(dim=1)
        elif self.pooling == "attn":
            # attention pooling: compute scores between Fs and v_mod
            scores = (v_mod * Fs).sum(dim=-1)  # [B,N]
            attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B,N,1]
            pooled = (v_mod * attn).sum(dim=1)  # [B,C]
        else:
            pooled = v_mod.mean(dim=1)

        vlv_cls = self.pool_proj(pooled).unsqueeze(1)  # [B,1,C]

        return vlv_cls, S  # S: [B,N]
