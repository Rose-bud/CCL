import torch.nn as nn
import torch


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class simpleNet(nn.Module):
    def __init__(self, input_dim):
        super(simpleNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ExpertHead(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ExpertHead, self).__init__()
        self.expertHead = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])

    def forward(self, x_chunk, gate_head):
        expert_outputs = [expert(x_chunk[i]) for i, expert in enumerate(self.expertHead)]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = expert_outputs * gate_head.squeeze(1).unsqueeze(2)
        return expert_outputs


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12,  num_experts=7, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_experts = num_experts
        self.linear_re = nn.Sequential(
            nn.Linear(num_experts * dim, dim),
            QuickGELU(),
            nn.BatchNorm1d(dim),
        )
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        B, N, C = y.shape  # N = num_experts
        x = self.linear_re(x)
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        gates = attn.softmax(dim=-1)
        return gates

    def forward_(self, x):
        x = self.direct_gate(x)
        return x.unsqueeze(1)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, head, num_experts=7):
        super(GatingNetwork, self).__init__()
        self.gate = CrossAttention(input_dim, head, num_experts=num_experts)

    def forward(self, x, y):
        gates = self.gate(x, y)
        return gates


class MoM(nn.Module):
    def __init__(self, input_dim, num_experts, head):
        super(MoM, self).__init__()
        self.head_dim = input_dim // head
        self.head = head
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ExpertHead(self.head_dim, num_experts) for _ in range(head)])
        self.gating_network = GatingNetwork(input_dim, head, num_experts=num_experts)

    def forward(self, *xs):
        # xs: list of expert input features (num_experts)
        num_experts = len(xs)
        if xs[0].dim() == 1:
            xs = [x.unsqueeze(0) for x in xs]

        # chunk by head
        chunks = [torch.chunk(x, self.head, dim=-1) for x in xs]  # list of len(num_experts), each [head, B, dim//head]
        head_input = [[chunks[j][i] for j in range(num_experts)] for i in range(self.head)]
        query = torch.cat(xs, dim=-1)
        key = torch.stack(xs, dim=1)
        gate_heads = self.gating_network(query, key)
        expert_outputs = [expert(head_input[i], gate_heads[:, i]) for i, expert in enumerate(self.experts)]
        outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)
        loss = 0
        if self.training:
            return outputs, loss
        return outputs


class GeneralFusion(nn.Module):
    def __init__(self, feat_dim, num_experts, head, reg_weight=0.1, dropout=0.1, cfg=None):
        super(GeneralFusion, self).__init__()
        self.reg_weight = reg_weight
        self.feat_dim = feat_dim

        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        if self.HDM:
            self.dropout = dropout
            scale = self.feat_dim ** -0.5
            self.r_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.n_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.t_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rn_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.nt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rnt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            head_num_attn = self.feat_dim // 64
            self.r = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.n = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.t = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.nt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rnt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
        if self.ATM:
            self.moe = MoM(input_dim=self.feat_dim, num_experts=num_experts, head=head)


    def forward_HDM(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)
        batch = RGB.size(1)
        # get the learnable token
        r_embedding = self.r_token.repeat(1, batch, 1)
        n_embedding = self.n_token.repeat(1, batch, 1)
        t_embedding = self.t_token.repeat(1, batch, 1)
        rn_embedding = self.rn_token.repeat(1, batch, 1)
        rt_embedding = self.rt_token.repeat(1, batch, 1)
        nt_embedding = self.nt_token.repeat(1, batch, 1)
        rnt_embedding = self.rnt_token.repeat(1, batch, 1)

        # for single modality
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze(-1)
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze(-1)
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze(-1)
        RN_shared  = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze(-1)
        RT_shared  = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze(-1)
        NT_shared  = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze(-1)
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze(-1)

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared

    def forward_ATM(self, RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared,*experts_list):
        if self.training:
            moe_feat, loss_reg = self.moe(*experts_list)
            return moe_feat, self.reg_weight * loss_reg
        else:
            moe_feat = self.moe(*experts_list)
            return moe_feat

    def forward(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global, extra_expert_tokens=None):
        RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDM(
            RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)

        if extra_expert_tokens is not None:
            # extra_expert_tokens expected shape [B, 3, C]
            RGB_vlv, NI_vlv, TI_vlv = torch.unbind(extra_expert_tokens, dim=1)
            experts_list = [RGB_vlv, NI_vlv, TI_vlv, RGB_special, NI_special, TI_special,
                            RN_shared, RT_shared, NT_shared, RNT_shared]
        else:
            experts_list = [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared]
        if self.training:
            if self.HDM and not self.ATM:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared], dim=-1)
                return moe_feat, 0
            elif self.HDM and self.ATM:
                moe_feat, loss_reg = self.forward_ATM(RGB_special, NI_special, TI_special, RN_shared, RT_shared,
                                                      NT_shared, RNT_shared, *experts_list)
                return moe_feat, loss_reg
        else:
            if self.HDM and not self.ATM:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared], dim=-1)
                if moe_feat.dim() == 1:
                    moe_feat = moe_feat.unsqueeze(0)
                return moe_feat
            elif self.HDM and self.ATM:
                moe_feat = self.forward_ATM(RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared,
                                            RNT_shared, *experts_list)
                return moe_feat
