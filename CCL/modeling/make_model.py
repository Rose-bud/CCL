import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.meta_arch_text import build_transformer as build_transformer_text
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU
import torch
from modeling.clip import clip
from modeling.rgb_vlv_module import RGBVisualLinguisticVerification as VisualLinguisticVerificationVLTVG

class CCL(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(CCL, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.num_classes = num_classes
        self.cfg = cfg
        self.isText = cfg.MODEL.isText
        if self.isText:
            self.BACKBONE = build_transformer_text(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        else:
            self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        num_experts = 7
        if self.isText:
            self.vlv_rgb = VisualLinguisticVerificationVLTVG(feat_dim=self.feat_dim, text_dim=self.feat_dim,cfg=cfg)
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        if self.GLOBAL_LOCAL:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
            self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
            self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=num_experts, head=self.head,
                                               cfg=cfg)
            self.classifier_moe = nn.Linear(num_experts * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(num_experts * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)
        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)
        self.current_epoch_weights = None
        self.warmup = cfg.SOLVER.SPL_WARMUP

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_n = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_t = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        cam_label = torch.tensor(0, device=next(model.parameters()).device, dtype=torch.int64)
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label,
                 'text': {'rgb_text': clip.tokenize('just a test').cuda(),
                          'ni_text': clip.tokenize('just a test').cuda(),
                          'ti_text': clip.tokenize('just a test').cuda()}}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "For the Mamba Series, the code implementations are all used with the inner weight instead of directly calling the model, the FLOPs has been calculated with our inner function 'MambaInnerFn_jit', so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, text=None,label=None, cam_label=None, view_label=None, return_pattern=3, epoch=1,img_path=None,ppp=1):
        if 'cam_label' in x:
            cam_label = x['cam_label']
        if 'text' in x:
            RGB_Text = x['text']['rgb_text']
            NI_Text = x['text']['ni_text']
            TI_Text = x['text']['ti_text']
        else:
            RGB_Text = text['rgb_text']
            NI_Text = text['ni_text']
            TI_Text = text['ti_text']
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']
        if self.training:
            if self.isText:
                RGB_cash, RGB_global, RGB_t_feas, RGB_t_global = self.BACKBONE(RGB, text=RGB_Text,cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global, NI_t_feas, NI_t_global = self.BACKBONE(NI, text=NI_Text,cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global, TI_t_feas, TI_t_global= self.BACKBONE(TI, text=TI_Text,cam_label=cam_label, view_label=view_label)
                sample_weights = None
                if epoch > self.warmup:
                    sample_weights = getattr(self, 'current_epoch_weights', None)
                    if isinstance(sample_weights, dict) and sample_weights:
                        sample_ids = []
                        if img_path is not None:
                            try:
                                if isinstance(img_path, (list, tuple)):
                                    sample_ids = [str(name) for name in img_path]
                                else:
                                    sample_ids = [str(img_path[i].item()) if hasattr(img_path[i], 'item') else str(img_path[i]) for i in range(len(img_path))]
                            except Exception:
                                sample_ids = []
                        
                        if sample_ids:
                            weights_list = [
                                sample_weights.get(sample_id, 1.0)
                                for sample_id in sample_ids
                            ]
                            sample_weights = torch.tensor(weights_list, device=RGB_cash.device)
                        else:
                            sample_weights = torch.ones(RGB_cash.size(0), device=RGB_cash.device)
                    elif sample_weights is None:
                        sample_weights = torch.ones(RGB_cash.size(0), device=RGB_cash.device)
                    
                _, _, rgb_all, _ = self.vlv_rgb(RGB_cash, RGB_t_feas, sample_weights=sample_weights,epoch=epoch,ppp=ppp)
                _, _, nir_all, _ = self.vlv_rgb(NI_cash, NI_t_feas, sample_weights=sample_weights,epoch=epoch,ppp=ppp)
                _, _, tir_all, _ = self.vlv_rgb(TI_cash, TI_t_feas, sample_weights=sample_weights,epoch=epoch,ppp=ppp)
                vlv_tokens=None
            else:
                RGB_cash, RGB_global = self.BACKBONE(RGB,cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.BACKBONE(NI,cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE(TI,cam_label=cam_label, view_label=view_label)
            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
            if self.HDM or self.ATM:    
                if self.isText:
                    moe_feat , loss_moe = self.generalFusion(rgb_all,nir_all, tir_all,
                                            RGB_global, NI_global, TI_global,
                                            extra_expert_tokens=vlv_tokens)
                else:
                    moe_feat, loss_moe  = self.generalFusion(RGB_cash, NI_cash, TI_cash,
                                            RGB_global, NI_global, TI_global)
                moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
            if self.direct:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
            else:
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))
            if self.direct:
                if self.HDM or self.ATM:
                    return moe_score, moe_feat, ori_score, ori, loss_moe
                return ori_score, ori
            else:
                if self.HDM or self.ATM:
                    return moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe
                return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
                if self.isText:
                    RGB_Text = torch.zeros_like(RGB_Text)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
                if self.isText:
                    NI_Text = torch.zeros_like(NI_Text)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
                if self.isText:
                    TI_Text = torch.zeros_like(TI_Text)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
                if self.isText:
                    RGB_Text = torch.zeros_like(RGB_Text)
                    NI_Text = torch.zeros_like(NI_Text)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
                if self.isText:
                    RGB_Text = torch.zeros_like(RGB_Text)
                    TI_Text = torch.zeros_like(TI_Text)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)
                if self.isText:
                    NI_Text = torch.zeros_like(NI_Text)
                    TI_Text = torch.zeros_like(TI_Text)

            if 'cam_label' in x:
                cam_label = x['cam_label']
            if self.isText:
                RGB_cash, RGB_global, RGB_t_feas, RGB_t_global = self.BACKBONE(RGB, text=RGB_Text,cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global, NI_t_feas, NI_t_global = self.BACKBONE(NI, text=NI_Text,cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global, TI_t_feas, TI_t_global= self.BACKBONE(TI, text=TI_Text,cam_label=cam_label, view_label=view_label)
                _, _, rgb_all, _ = self.vlv_rgb(RGB_cash, RGB_t_feas,ppp=ppp)
                _, _, nir_all, _ = self.vlv_rgb(NI_cash, NI_t_feas,ppp=ppp)
                _, _, tir_all, _ = self.vlv_rgb(TI_cash, TI_t_feas,ppp=ppp)
                vlv_tokens=None
            else:
                RGB_cash, RGB_global = self.BACKBONE(RGB,cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.BACKBONE(NI,cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE(TI,cam_label=cam_label, view_label=view_label)
                vlv_tokens=None
            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            if self.HDM or self.ATM:
                if self.isText:
                    moe_feat = self.generalFusion(rgb_all,nir_all, tir_all,
                                            RGB_global, NI_global, TI_global,
                                            extra_expert_tokens=vlv_tokens)
                else:
                    moe_feat = self.generalFusion(RGB_cash, NI_cash, TI_cash,
                                        RGB_global, NI_global, TI_global)
                if return_pattern == 1:
                    return ori
                elif return_pattern == 2:
                    return moe_feat
                elif return_pattern == 3:
                    return torch.cat([ori, moe_feat], dim=-1)
            return ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = CCL(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building CCL===========')
    return model
