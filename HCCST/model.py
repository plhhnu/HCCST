import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align
from torch_geometric.nn import GATConv

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimpleCrossModalContrastiveLoss(nn.Module):
    def __init__(self, embed_dim: int, temperature: float = 0.07, learnable_temp: bool = True, eps: float = 1e-12):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        if learnable_temp:
            init_scale = 1.0 / max(temperature, 1e-6)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.register_buffer(
                "logit_scale",
                torch.log(torch.tensor(1.0 / max(temperature, 1e-6), dtype=torch.float32)),
            )
    def forward(self, z_a, z_b, pos_idx=None, pos_mask=None):
        device = z_a.device
        B = z_a.size(0)
        za = l2_normalize(z_a, self.eps)
        zb = l2_normalize(z_b, self.eps)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = torch.matmul(za, zb.t()) * scale
        if pos_idx is None:
            pos_idx = torch.arange(B, device=device)
        eye = torch.zeros(B, B, dtype=torch.bool, device=device)
        eye[torch.arange(B, device=device), pos_idx] = True
        if pos_mask is not None:
            pos_mask = pos_mask.to(device).bool()
            # Keep each paired positive visible while removing spatial neighbors from negatives.
            allowed_mask = eye | (~pos_mask)
            fill_value = -1e4 if logits.dtype == torch.float16 else -1e9
            logits_g2i = logits.masked_fill(~allowed_mask, fill_value)
            logits_i2g = logits.t().masked_fill(~allowed_mask.t(), fill_value)
        else:
            logits_g2i = logits
            logits_i2g = logits.t()
        targets = pos_idx
        loss_g2i = F.cross_entropy(logits_g2i, targets)
        loss_i2g = F.cross_entropy(logits_i2g, targets)
        loss = 0.5 * (loss_g2i + loss_i2g)
        with torch.no_grad():
            sim_raw = torch.matmul(za, zb.t())
            s_pos_mean = sim_raw[torch.arange(B, device=device), pos_idx].mean()
            if pos_mask is not None:
                neg_mask = (~pos_mask) & (~eye)
            else:
                neg_mask = ~eye
            s_neg_mean = sim_raw[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=device)
            var_a = za.var(dim=0).mean()
            var_b = zb.var(dim=0).mean()
        return {
            "loss": loss,
            "raw_loss": loss.detach(),
            "weight": torch.tensor(1.0, device=device),
            "temperature": (1.0 / scale).detach(),
            "s_pos_mean": s_pos_mean.detach(),
            "s_neg_topk_mean": s_neg_mean.detach(),
            "var_a": var_a.detach(),
            "var_b": var_b.detach(),
        }

class HistologyEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, output_dim)
        )
        self.projection_2048 = self.projection
        self.projection_1024 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, output_dim)
        )
        self.projection_512 = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = self.adaptive_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        projected = self.projection(flattened)
        return projected, features
    def _normalize_roi_boxes(self, roi_boxes_per_spot):
        if roi_boxes_per_spot.dim() == 2:
            return roi_boxes_per_spot.size(0), 1, roi_boxes_per_spot.unsqueeze(1)
        if roi_boxes_per_spot.dim() == 3:
            n, t, _ = roi_boxes_per_spot.shape
            return n, t, roi_boxes_per_spot
        raise ValueError(f"Unsupported ROI box dimensions: {roi_boxes_per_spot.dim()}")
    def _projector_for_channels(self, channels):
        if channels == 2048:
            return self.projection_2048
        if channels == 1024:
            return self.projection_1024
        if channels == 512:
            return self.projection_512
        raise ValueError(f"Unsupported feature channel count: {channels}")
    def _roi_align_project(self, feature_map, boxes, image_size, output_size=1):
        feat_h, feat_w = feature_map.size(-2), feature_map.size(-1)
        img_w, img_h = map(float, image_size)
        ratio_w = torch.tensor(feat_w / img_w, dtype=torch.float64, device=feature_map.device)
        ratio_h = torch.tensor(feat_h / img_h, dtype=torch.float64, device=feature_map.device)
        boxes_q = torch.round(boxes.to(feature_map.device) * 1000) / 1000
        sampling_ratio, eps = 2, 1e-8
        if abs(ratio_w - ratio_h) < eps:
            pooled = roi_align(
                feature_map,
                [boxes_q.float()],
                output_size=(output_size, output_size),
                spatial_scale=float(ratio_w),
                sampling_ratio=sampling_ratio,
                aligned=True,
            )
        else:
            # Non-square resize ratios require projecting ROI coordinates manually.
            boxes_feat = boxes_q.clone().double()
            boxes_feat[:, [0,2]] *= ratio_w
            boxes_feat[:, [1,3]] *= ratio_h
            boxes_feat = torch.round(boxes_feat * 1000) / 1000
            pooled = roi_align(
                feature_map,
                [boxes_feat.float()],
                output_size=(output_size, output_size),
                spatial_scale=1.0,
                sampling_ratio=sampling_ratio,
                aligned=True,
            )
        if output_size == 1:
            pooled = pooled.squeeze(-1).squeeze(-1)
        else:
            pooled = F.adaptive_avg_pool2d(pooled, (1,1)).squeeze(-1).squeeze(-1)
        pooled = F.normalize(pooled, p=2, dim=1)
        return self._projector_for_channels(feature_map.size(1))(pooled)
    def _scale_boxes(self, flat_boxes, image_size, scales):
        W, H = image_size
        cx = (flat_boxes[:,0] + flat_boxes[:,2]) * 0.5
        cy = (flat_boxes[:,1] + flat_boxes[:,3]) * 0.5
        w = (flat_boxes[:,2] - flat_boxes[:,0]).clamp(min=1.0)
        h = (flat_boxes[:,3] - flat_boxes[:,1]).clamp(min=1.0)
        scaled_boxes = []
        for s in scales:
            sw, sh = w * float(s), h * float(s)
            scaled_boxes.append(torch.stack([
                (cx - sw*0.5).clamp(min=0.0, max=float(W-1)),
                (cy - sh*0.5).clamp(min=0.0, max=float(H-1)),
                (cx + sw*0.5).clamp(min=0.0, max=float(W-1)),
                (cy + sh*0.5).clamp(min=0.0, max=float(H-1)),
            ], dim=1))
        return torch.cat(scaled_boxes, dim=0)
    def extract_patch_tokens(self, feature_map, roi_boxes_per_spot, image_size, output_size=1):
        assert feature_map.dim() == 4 and feature_map.size(0) == 1
        N, T, roi_boxes_per_spot = self._normalize_roi_boxes(roi_boxes_per_spot)
        token_embeds = self._roi_align_project(
            feature_map,
            roi_boxes_per_spot.reshape(-1, 4),
            image_size,
            output_size=output_size,
        )
        return token_embeds.reshape(N, T, -1)

    def extract_multi_scale_tokens(self, feature_map, roi_boxes_per_spot, image_size, scales=(1.0, 2.0), output_size=1):
        assert feature_map.dim() == 4 and feature_map.size(0) == 1
        N, T, roi_boxes_per_spot = self._normalize_roi_boxes(roi_boxes_per_spot)
        flat_boxes = roi_boxes_per_spot.reshape(-1, 4).to(feature_map.device)
        all_boxes = self._scale_boxes(flat_boxes, image_size, scales)
        token_embeds = self._roi_align_project(feature_map, all_boxes, image_size, output_size=output_size)
        return token_embeds.reshape(N, T * len(scales), -1)

    def extract_fpn_feature_maps(self, x):
        y = self.backbone.conv1(x)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)
        c2 = self.backbone.layer1(y)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return c3, c4, c5
    def extract_pyramid_multi_scale_tokens(
        self,
        image,
        roi_boxes_per_spot,
        image_size,
        scales=(1.0, 2.0),
        output_size=1,
    ):
        c3, c4, c5 = self.extract_fpn_feature_maps(image)
        tokens_list = []
        for fmap in (c3, c4, c5):
            t = self.extract_multi_scale_tokens(fmap, roi_boxes_per_spot, image_size, scales, output_size)
            tokens_list.append(t)
        tokens = torch.cat(tokens_list, dim=1)
        return tokens

class GeneExpressionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1, fill_value='mean')
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, edge_dim=1, fill_value='mean')
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.pre_train_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, pre_train=False):
        x = self.gat1(x, edge_index, edge_attr=edge_weight)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = F.elu(self.bn2(x))
        if pre_train:
            recon = self.pre_train_head(x)
            return x, recon
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, gene_dim, image_dim, hidden_dim=512, num_heads=8, config=None):
        super().__init__()
        self.config = config or {}
        self.gene2img_attn = nn.MultiheadAttention(gene_dim, num_heads, kdim=image_dim, vdim=image_dim)
        if not self.config.get('disable_bidirectional_attention', False):
            self.img2gene_attn = nn.MultiheadAttention(image_dim, num_heads, kdim=gene_dim, vdim=gene_dim)
        else:
            self.img2gene_attn = None
        if not self.config.get('disable_gated_fusion', False):
            self.gate = nn.Sequential(
                nn.Linear(gene_dim + image_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, gene_dim + image_dim),
                nn.Sigmoid()
            )
        else:
            self.gate = None
        self.norm1 = nn.LayerNorm(gene_dim)
        self.norm2 = nn.LayerNorm(image_dim)

    def forward(self, gene_features, image_features=None, image_tokens=None):
        if image_tokens is not None:
            q_gene = gene_features.unsqueeze(0).contiguous()
            kvi = image_tokens.permute(1, 0, 2).contiguous()
            attn_g2i, _ = self.gene2img_attn(q_gene, kvi, kvi)
            gene_enhanced = attn_g2i.squeeze(0)
        else:
            assert image_features is not None
            q_gene = gene_features.unsqueeze(1).contiguous()
            kvi = image_features.unsqueeze(1).contiguous()
            attn_g2i, _ = self.gene2img_attn(q_gene, kvi, kvi)
            gene_enhanced = attn_g2i.squeeze(1)
        if self.img2gene_attn is not None:
            if image_tokens is not None:
                q_img = kvi
                kvg_gene = gene_features.unsqueeze(0).contiguous()
                attn_i2g, _ = self.img2gene_attn(q_img, kvg_gene, kvg_gene)
                image_enhanced = attn_i2g.mean(dim=0)
                image_base = image_features if image_features is not None else image_enhanced
            else:
                q_img = kvi
                kvg_gene = gene_features.unsqueeze(1).contiguous()
                attn_i2g, _ = self.img2gene_attn(q_img, kvg_gene, kvg_gene)
                image_enhanced = attn_i2g.squeeze(1)
                image_base = image_features
        else:
            image_enhanced = torch.zeros_like(
                image_features if image_features is not None else image_tokens.mean(dim=1)
            )
            image_base = image_features if image_features is not None else image_tokens.mean(dim=1)
        gene_out = self.norm1(gene_features + gene_enhanced)
        image_out = self.norm2(image_base + image_enhanced)
        if self.gate is not None:
            combined = torch.cat([gene_out, image_out], dim=1)
            gate_weights = self.gate(combined)
            gene_gate, image_gate = torch.split(gate_weights, [gene_out.size(1), image_out.size(1)], dim=1)
            fused_gene = gene_gate * gene_out
            fused_image = image_gate * image_out
        else:
            fused_gene = gene_out
            fused_image = image_out
        gene_bal = F.normalize(fused_gene, p=2, dim=1)
        image_bal = F.normalize(fused_image, p=2, dim=1)
        fused_features = torch.cat([gene_bal, image_bal], dim=1)
        return fused_features, gene_out, image_out, fused_gene, fused_image, {}

class CellTypePredictor(nn.Module):
    def __init__(self, input_dim, num_cell_types, hidden_dims=[512,256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, num_cell_types))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=1), logits

class MultimodalLoss(nn.Module):
    """Compute the four losses used by HCCST.

    The total objective is assembled in ``HCCST.py`` as:
    alpha * L_recon + beta * L_con + gamma * L_cons + delta * L_proto.
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.7, delta=0.1, consistency_side="gene"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.consistency_side = consistency_side
        self.mse_loss = nn.MSELoss()
    def reconstruction_loss(self, gene_recon, gene_target):
        return self.mse_loss(gene_recon, gene_target)
    def consistency_loss(self, gene_features, image_features, fused_gene, fused_image):
        eps = 1e-8
        gene_target = F.normalize(gene_features.detach(), p=2, dim=1, eps=eps)
        image_target = F.normalize(image_features.detach(), p=2, dim=1, eps=eps)
        fused_gene_norm = F.normalize(fused_gene, p=2, dim=1, eps=eps)
        fused_image_norm = F.normalize(fused_image, p=2, dim=1, eps=eps)
        gene_loss_per = 1.0 - F.cosine_similarity(fused_gene_norm, gene_target, dim=1)
        image_loss_per = 1.0 - F.cosine_similarity(fused_image_norm, image_target, dim=1)
        denom = gene_loss_per.numel() + eps
        if self.consistency_side == "gene":
            loss = gene_loss_per.sum() / denom
        elif self.consistency_side == "image":
            loss = image_loss_per.sum() / denom
        else:
            loss_gene = gene_loss_per.sum() / denom
            loss_image = image_loss_per.sum() / denom
            loss = 0.5 * (loss_gene + loss_image)
        return loss
    def prototype_loss(self, fused_gene, fused_image):
        if self.delta <= 0:
            return fused_gene.new_tensor(0.0)
        zg = F.normalize(fused_gene, p=2, dim=1)
        zi = F.normalize(fused_image, p=2, dim=1)
        proto_g = zg.mean(dim=0, keepdim=True)
        proto_i = zi.mean(dim=0, keepdim=True)
        proto_g = F.normalize(proto_g, p=2, dim=1)
        proto_i = F.normalize(proto_i, p=2, dim=1)
        if proto_g.size(1) != proto_i.size(1):
            loss = (1.0 - (zg @ proto_g.t()).mean()) + (1.0 - (zi @ proto_i.t()).mean())
            return 0.5 * loss
        return 1.0 - F.cosine_similarity(proto_g, proto_i, dim=1).mean()
    def forward(self, gene_recon, gene_target, gene_features, image_features,
                fused_gene, fused_image, **kwargs):
        recon = self.reconstruction_loss(gene_recon, gene_target)
        consistency = self.consistency_loss(gene_features, image_features, fused_gene, fused_image)
        prototype = self.prototype_loss(fused_gene, fused_image)
        return {"recon": recon, "consistency": consistency, "prototype": prototype}

class MultimodalIntegrationModel(nn.Module):
    def __init__(self, gene_input_dim, num_cell_types, gene_hidden_dim=512, gene_output_dim=64,
                 image_output_dim=64, cross_attn_heads=8, config=None, morph_input_dim=None):
        super().__init__()
        # Only keep switches that define the current main experiment path.
        self.config = {
            'disable_image': False,
            'disable_cross_attention': False,
            'disable_roi_tokens': False,
            'disable_projection_head': False,
            'disable_contrastive': False,
            'disable_bidirectional_attention': False,
            'disable_gated_fusion': False,
            'enable_multi_scale_roi': True,
            'multi_scale_scales': (1.0, 2.0),
            'multi_scale_fuse_mode': 'mean',
            'use_precomputed_morphology': False,
        }
        if config:
            self.config.update(config)
        self.gene_encoder = GeneExpressionEncoder(gene_input_dim, gene_hidden_dim, gene_output_dim)
        self.image_encoder = HistologyEncoder(output_dim=image_output_dim)
        if morph_input_dim is not None:
            self.morph_feature_projector = nn.Sequential(
                nn.Linear(int(morph_input_dim), image_output_dim),
                nn.LayerNorm(image_output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        self.gene_output_dim = gene_output_dim
        self.image_output_dim = image_output_dim
        self.gene_to_image_bridge = nn.Linear(gene_output_dim, image_output_dim)
        self.scale_aggregator = nn.MultiheadAttention(image_output_dim, cross_attn_heads)
        self.cross_attention = CrossModalAttention(gene_output_dim, image_output_dim, config=self.config)
        self.cell_predictor = CellTypePredictor(gene_output_dim + image_output_dim, num_cell_types)
        self.embed_dim = (
            min(gene_output_dim, image_output_dim)
            if gene_output_dim != image_output_dim
            else image_output_dim
        )
        if self.config.get('disable_projection_head', False):
            self.gene_proj = nn.Identity()
            self.image_proj = nn.Identity()
        else:
            self.gene_proj = ProjectionHead(gene_output_dim, self.embed_dim)
            self.image_proj = ProjectionHead(image_output_dim, self.embed_dim)
        self.cross_modal_loss = (
            None
            if self.config.get('disable_contrastive', False)
            else SimpleCrossModalContrastiveLoss(self.embed_dim)
        )

    def _prepare_precomputed_morphology(self, image_data, gene_features):
        morph = torch.as_tensor(
            image_data['morph_features'],
            dtype=torch.float32,
            device=gene_features.device,
        )
        if morph.ndim != 2 or morph.size(0) != gene_features.size(0):
            raise ValueError(
                "morph_features must have shape [n_spots, d_morph] "
                f"with n_spots={gene_features.size(0)}, got {tuple(morph.shape)}"
            )
        projector = getattr(self, 'morph_feature_projector', None)
        if projector is None:
            if morph.size(1) != self.image_output_dim:
                raise ValueError("morph_input_dim is required for precomputed morphology features")
            return morph
        return projector(morph)

    def _fuse_precomputed_morphology(self, gene_features, image_features):
        if self.config.get('disable_cross_attention', False):
            fused_gene, fused_image = gene_features, image_features
            fused_features = torch.cat(
                [F.normalize(fused_gene, p=2, dim=1), F.normalize(fused_image, p=2, dim=1)],
                dim=1,
            )
            return fused_features, fused_gene, fused_image, fused_gene, fused_image, {}
        return self.cross_attention(
            gene_features,
            image_features=image_features,
            image_tokens=None,
        )

    def forward(self, gene_data, image_data, edge_index=None, edge_weight=None, pre_train=False):
        if pre_train:
            gene_features, gene_recon = self.gene_encoder(
                gene_data,
                edge_index,
                edge_weight=edge_weight,
                pre_train=True,
            )
            return gene_features, gene_recon, None, None, None, None, None, None, None
        gene_features = self.gene_encoder(gene_data, edge_index, edge_weight=edge_weight)
        if self.config.get('disable_image', False) or (image_data is None):
            gene_enhanced = gene_features
            fused_gene = gene_enhanced
            image_enhanced = torch.zeros(gene_features.size(0), self.image_output_dim, device=gene_features.device)
            fused_image = image_enhanced
            fused_features = torch.cat(
                [F.normalize(fused_gene, p=2, dim=1), F.normalize(fused_image, p=2, dim=1)],
                dim=1,
            )
            _, cell_logits = self.cell_predictor(fused_features)
            aux_stats = {}
            return (
                gene_features,
                gene_enhanced,
                image_enhanced,
                fused_features,
                fused_gene,
                fused_image,
                cell_logits,
                aux_stats,
            )
        if (
            self.config.get('use_precomputed_morphology', False)
            and isinstance(image_data, dict)
            and 'morph_features' in image_data
        ):
            image_features = self._prepare_precomputed_morphology(image_data, gene_features)
            fused_features, gene_enhanced, image_enhanced, fused_gene, fused_image, aux_stats = (
                self._fuse_precomputed_morphology(gene_features, image_features)
            )
            _, cell_logits = self.cell_predictor(fused_features)
            return (
                gene_features,
                gene_enhanced,
                image_enhanced,
                fused_features,
                fused_gene,
                fused_image,
                cell_logits,
                aux_stats,
            )
        if isinstance(image_data, dict):
            image_tensor = image_data['image']
            global_image_feat, feat_map = self.image_encoder(image_tensor)
            image_tokens = None
            image_feat_for_cross = global_image_feat.expand(gene_features.size(0), -1).contiguous()
            if (
                (not self.config.get('disable_roi_tokens', False))
                and ('roi_boxes' in image_data)
                and (image_data['roi_boxes'] is not None)
            ):
                if self.config.get('enable_multi_scale_roi', True):
                    # Pyramid ROI tokens combine local morphology from multiple ResNet stages.
                    image_tokens = self.image_encoder.extract_pyramid_multi_scale_tokens(
                        image_tensor,
                        image_data['roi_boxes'],
                        image_data.get('image_size', (image_tensor.size(3), image_tensor.size(2))),
                        scales=self.config.get('multi_scale_scales', (1.0, 2.0)),
                        output_size=1,
                    )
                else:
                    image_tokens = self.image_encoder.extract_patch_tokens(
                        feat_map,
                        image_data['roi_boxes'],
                        image_data.get('image_size', (image_tensor.size(3), image_tensor.size(2))),
                        output_size=1,
                    )
                fuse_mode = self.config.get('multi_scale_fuse_mode', 'mean')
                if image_tokens is not None:
                    if fuse_mode == 'attn':
                        q = self.gene_to_image_bridge(gene_features).unsqueeze(0)
                        kv = image_tokens.permute(1,0,2).contiguous()
                        agg, _ = self.scale_aggregator(q, kv, kv)
                        image_feat_for_cross = agg.squeeze(0)
                    elif fuse_mode == 'max':
                        image_feat_for_cross = image_tokens.max(dim=1).values
                    else:
                        image_feat_for_cross = image_tokens.mean(dim=1)
            if self.config.get('disable_cross_attention', False):
                fused_gene = gene_features
                fused_image = image_feat_for_cross
                gene_enhanced = fused_gene
                image_enhanced = fused_image
                fused_features = torch.cat(
                    [F.normalize(fused_gene, p=2, dim=1), F.normalize(fused_image, p=2, dim=1)],
                    dim=1,
                )
                aux_stats = {}
            else:
                fused_features, gene_enhanced, image_enhanced, fused_gene, fused_image, aux_stats = (
                    self.cross_attention(
                        gene_features,
                        image_features=image_feat_for_cross,
                        image_tokens=image_tokens,
                    )
                )
        else:
            image_features, _ = self.image_encoder(image_data)
            if image_features.size(0) == 1 and gene_features.size(0) > 1:
                image_features = image_features.expand(gene_features.size(0), -1).contiguous()
            if self.config.get('disable_cross_attention', False):
                fused_features = torch.cat([gene_features, image_features], dim=1)
                gene_enhanced = gene_features
                image_enhanced = image_features
                fused_gene = gene_enhanced
                fused_image = image_enhanced
                aux_stats = {}
            else:
                fused_features, gene_enhanced, image_enhanced, fused_gene, fused_image, aux_stats = (
                    self.cross_attention(gene_features, image_features=image_features)
                )
        _, cell_logits = self.cell_predictor(fused_features)
        return (
            gene_features,
            gene_enhanced,
            image_enhanced,
            fused_features,
            fused_gene,
            fused_image,
            cell_logits,
            aux_stats,
        )

    def compute_cross_modal_loss(self, gene_emb, image_emb, pos_mask=None):
        degenerate = bool(self.config.get('disable_roi_tokens', False))
        low_var = False
        if isinstance(image_emb, torch.Tensor) and image_emb.ndim == 2 and image_emb.size(0) > 1:
            with torch.no_grad():
                low_var = (image_emb.float().std(dim=0, unbiased=False).mean() < 1e-6)
        # Return a differentiable zero for disabled or degenerate-image cases.
        if (
            (self.cross_modal_loss is None)
            or self.config.get('disable_contrastive', False)
            or degenerate
            or low_var
            or (gene_emb is None)
            or (image_emb is None)
        ):
            device = gene_emb.device if isinstance(gene_emb, torch.Tensor) else image_emb.device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'loss': zero,
                'raw_loss': zero.detach(),
                'weight': torch.tensor(0.0, device=device),
                'temperature': torch.tensor(0.0, device=device),
                's_pos_mean': zero,
                's_neg_topk_mean': zero,
                'var_a': zero,
                'var_b': zero,
            }
        g = self.gene_proj(gene_emb)
        i = self.image_proj(image_emb)
        pos_idx = torch.arange(g.shape[0], device=g.device)
        return self.cross_modal_loss(g, i, pos_idx=pos_idx, pos_mask=pos_mask)
