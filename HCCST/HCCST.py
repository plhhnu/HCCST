import torch
import numpy as np
from .preprocess import (
    construct_interaction,
    construct_interaction_KNN,
    fix_seed,
    get_feature,
    preprocess,
    preprocess_adj,
    preprocess_adj_sparse,
)
from .model import MultimodalLoss, MultimodalIntegrationModel
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from torch.cuda.amp import GradScaler, autocast
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .utils import clustering
import scipy.sparse as sp

def make_roi_boxes_from_adata(adata, patch_size, image_size, library_id=None, spatial_key: str = 'spatial'):
    """
    Map spatial coordinates to resized image pixels and build one ROI box per spot.
    """
    spatial_info = adata.uns.get('spatial', {})
    if library_id is None and len(spatial_info) > 0:
        library_id = list(spatial_info.keys())[0]
    if library_id is None:
        raise ValueError("Cannot determine library_id for image and scale metadata")

    lib = spatial_info[library_id]
    hires_img = lib['images'].get('hires', None)
    if hires_img is None:
        raise ValueError("Cannot find the hires image")
    hires_h, hires_w = hires_img.shape[0], hires_img.shape[1]
    scale_hires = float(lib.get('scalefactors', {}).get('tissue_hires_scalef', 1.0))

    target_w, target_h = int(image_size[0]), int(image_size[1])

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    y_raw = coords[:, 0]
    x_raw = coords[:, 1]

    need_downscale_to_hires = (x_raw.max() > hires_w) or (y_raw.max() > hires_h)
    if need_downscale_to_hires:
        x_hires = x_raw * scale_hires
        y_hires = y_raw * scale_hires
    else:
        x_hires = x_raw
        y_hires = y_raw

    sx = float(target_w) / float(hires_w)
    sy = float(target_h) / float(hires_h)
    x_img = x_hires * sx
    y_img = y_hires * sy

    # Center each ROI on a spot and clamp it to the resized image boundary.
    half = float(patch_size) / 2.0
    x1 = np.clip(x_img - half, 0.0, target_w - 1.0)
    y1 = np.clip(y_img - half, 0.0, target_h - 1.0)
    x2 = np.clip(x_img + half, 0.0, target_w - 1.0)
    y2 = np.clip(y_img + half, 0.0, target_h - 1.0)

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    roi_boxes = torch.from_numpy(boxes)
    return roi_boxes


class MultimodalHCCST:
    def __init__(self, 
        adata,
        image_data=None,
        device=torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=5e-4,
        epochs=600,
        dim_input=3000,
        dim_output=48,
        random_seed=42,
        alpha=8,
        beta=0.1,
        gamma=0.7,
        delta=0.1,
        num_cell_types=7,
        image_size=224,
        patch_size=112,
        datatype='10X',
        gene_hidden_dim=768,
        lr_gene=0.0005,
        pretrain_ratio=0.3,
        model_config=None,
        use_morphology_graph=True,
        morphology_pca_components=16,
        morphology_weight_clip_min=0.0,
        morphology_topk_keep=None,
        embedding_mode='late_fusion',
        embedding_lambda=0.1,
        simulated_data=False,
        morphology_feature_key='morph_features',
    ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.num_cell_types = num_cell_types
        self.datatype = datatype
        self.input_image_size = image_size
        self.patch_size = patch_size
        self.gene_hidden_dim = gene_hidden_dim

        self.lr_gene = lr_gene
        self.pretrain_ratio = pretrain_ratio
        self.model_config = model_config or {}
        self.use_morphology_graph = use_morphology_graph
        self.morphology_pca_components = morphology_pca_components
        self.morphology_weight_clip_min = morphology_weight_clip_min
        self.morphology_topk_keep = morphology_topk_keep
        self.embedding_mode = embedding_mode
        self.embedding_lambda = embedding_lambda
        self.simulated_data = bool(simulated_data)
        self.morphology_feature_key = morphology_feature_key

        fix_seed(self.random_seed)

        # Prepare features and the base spatial graph when they are not already present.
        if 'highly_variable' not in self.adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in self.adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'feat' not in self.adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)

        self.adj_key = 'mor_adj' if ('mor_adj' in self.adata.obsm and self.use_morphology_graph) else 'adj'
        self.adj_raw = self.adata.obsm[self.adj_key]

        self.dim_input = self.features.shape[1]
        self._refresh_graph_state()

        self.image_tensor = None
        self.image_size = None
        self.image_data = image_data
        if image_data is not None:
            if not (self.simulated_data and self._set_precomputed_morphology(image_data)):
                self.process_image_data(image_data)

        # Optionally reweight spatial edges with patch-level morphology similarity.
        has_precomputed_morphology = self._get_morphology_matrix() is not None
        if self.use_morphology_graph and (self.image_tensor is not None or has_precomputed_morphology):
            try:
                self._build_morphology_aware_graph()
                self.adj_key = 'mor_adj'
                self.adj_raw = self.adata.obsm[self.adj_key]
                self._refresh_graph_state()
            except Exception as e:
                print(f"[Warning] morphology-aware graph construction failed; falling back to the spatial graph: {e}")
                self.adj_key = 'adj'
                self.adj_raw = self.adata.obsm[self.adj_key]
                self._refresh_graph_state()

    def _set_precomputed_morphology(self, image_data):
        if not isinstance(image_data, dict):
            return False
        morph = image_data.get(self.morphology_feature_key)
        if morph is None:
            return False
        if torch.is_tensor(morph):
            morph = morph.detach().cpu().numpy()
        morph = np.asarray(morph, dtype=np.float32)
        if morph.ndim != 2 or morph.shape[0] != self.adata.n_obs:
            raise ValueError(
                "morph_features must have shape [n_spots, d_morph] "
                f"with n_spots={self.adata.n_obs}, got {morph.shape}"
            )
        self.adata.obsm[self.morphology_feature_key] = morph
        self.adata.uns['morphology_input_type'] = 'precomputed_feature_matrix'
        return True

    def _get_morphology_matrix(self):
        if not self.simulated_data or self.morphology_feature_key not in self.adata.obsm:
            return None
        return np.asarray(self.adata.obsm[self.morphology_feature_key], dtype=np.float32)

    def _make_model_image_data(self):
        morph = self._get_morphology_matrix()
        if morph is not None:
            return {'morph_features': torch.as_tensor(morph, dtype=torch.float32, device=self.device)}
        if self.image_tensor is None:
            return None
        H_img, W_img = self.image_tensor.shape[-2], self.image_tensor.shape[-1]
        return {
            'image': self.image_tensor,
            'roi_boxes': make_roi_boxes_from_adata(
                self.adata,
                patch_size=self.patch_size,
                image_size=(W_img, H_img),
            ),
            'image_size': (W_img, H_img),
        }

    def _refresh_graph_state(self):
        """Refresh graph tensors from the active adjacency matrix."""
        base = self.adata.obsm[self.adj_key]
        if sp.issparse(base):
            base = base.toarray()
        base = np.asarray(base, dtype=np.float32)
        self.adj_raw = base
        binary = (base > 0).astype(np.float32)
        self.graph_neigh = torch.FloatTensor(binary + np.eye(binary.shape[0], dtype=np.float32)).to(self.device)
        if self.datatype in ['Stereo', 'Slide']:
            self.adj = preprocess_adj_sparse(base).to(self.device)
        else:
            self.adj = torch.FloatTensor(preprocess_adj(base)).to(self.device)
        self.edge_index, self.edge_weight = self._prepare_edge_tensors(base)

    def _prepare_edge_tensors(self, adj_matrix):
        """Convert a dense or sparse adjacency matrix to PyG edge tensors."""
        if sp.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        rows, cols = np.nonzero(adj_matrix)
        weights = adj_matrix[rows, cols].astype(np.float32)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=self.device)
        edge_weight = torch.tensor(weights[:, None], dtype=torch.float32, device=self.device)
        return edge_index, edge_weight

    def _extract_patch_morphology_features(self, roi_boxes, pooled_size=8, batch_size=512):
        """Extract lightweight CPU morphology features from spot-level ROIs."""
        if self.image_tensor is None:
            raise ValueError('image_tensor is None')
        image_cpu = self.image_tensor.detach().cpu()
        boxes = roi_boxes.to('cpu', dtype=torch.float32)
        feats_all = []
        for start in range(0, boxes.shape[0], batch_size):
            cur = boxes[start:start + batch_size]
            batch_index = torch.zeros((cur.shape[0], 1), device='cpu', dtype=torch.float32)
            rois = torch.cat([batch_index, cur], dim=1)
            pooled = roi_align(
                image_cpu,
                rois,
                output_size=(pooled_size, pooled_size),
                spatial_scale=1.0,
                aligned=True
            )
            feats = pooled.flatten(start_dim=1)
            feats = F.normalize(feats, p=2, dim=1)
            feats_all.append(feats.numpy())
        return np.concatenate(feats_all, axis=0)

    def _build_morphology_aware_graph(self):
        """Build a morphology-aware adjacency by reweighting existing spatial edges."""
        if 'adj' not in self.adata.obsm:
            raise ValueError("adata.obsm['adj'] is required")
        image_feature = self._get_morphology_matrix()
        if image_feature is None:
            H_resized, W_resized = self.image_tensor.shape[-2], self.image_tensor.shape[-1]
            roi_boxes = make_roi_boxes_from_adata(
                self.adata,
                patch_size=self.patch_size,
                image_size=(W_resized, H_resized)
            )
            image_feature = self._extract_patch_morphology_features(roi_boxes)
        n_comp = int(min(self.morphology_pca_components, image_feature.shape[0], image_feature.shape[1]))
        n_comp = max(2, n_comp)
        image_feature_pca = PCA(
            n_components=n_comp,
            random_state=self.random_seed,
        ).fit_transform(image_feature).astype(np.float32)

        base_adj = self.adata.obsm['adj']
        if sp.issparse(base_adj):
            base_adj = base_adj.toarray()
        base_adj = np.asarray(base_adj, dtype=np.float32)
        rows, cols = np.nonzero(base_adj)

        feat = image_feature_pca
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        edge_sim = np.sum(feat_norm[rows] * feat_norm[cols], axis=1)
        edge_sim = np.nan_to_num(edge_sim, nan=0.0, posinf=0.0, neginf=0.0)
        edge_sim = np.clip(edge_sim, 0.0, 1.0)
        edge_sim[edge_sim < self.morphology_weight_clip_min] = 0.0

        mor_adj = np.zeros_like(base_adj, dtype=np.float32)
        mor_adj[rows, cols] = base_adj[rows, cols] * edge_sim

        if self.morphology_topk_keep is not None and int(self.morphology_topk_keep) > 0:
            k = int(self.morphology_topk_keep)
            pruned = np.zeros_like(mor_adj, dtype=np.float32)
            for i in range(mor_adj.shape[0]):
                idx = np.flatnonzero(mor_adj[i] > 0)
                if idx.size == 0:
                    continue
                keep = idx if idx.size <= k else idx[np.argsort(mor_adj[i, idx])[-k:]]
                pruned[i, keep] = mor_adj[i, keep]
            mor_adj = np.maximum(pruned, pruned.T)
        else:
            mor_adj = np.maximum(mor_adj, mor_adj.T)

        np.fill_diagonal(mor_adj, 1.0)
        self.adata.obsm['image_feature_pca'] = image_feature_pca
        self.adata.obsm['mor_adj'] = mor_adj.astype(np.float32)
        self.adata.uns['mor_adj_stats'] = {
            'pca_dim': int(n_comp),
            'nonzero_edges': int(np.count_nonzero(mor_adj)),
            'mean_weight': float(mor_adj[mor_adj > 0].mean()) if np.count_nonzero(mor_adj) > 0 else 0.0,
            'topk_keep': None if self.morphology_topk_keep is None else int(self.morphology_topk_keep),
        }
    def _select_embedding_for_clustering(self):
        """Select and cache the embedding used by downstream clustering."""
        mode = str(self.embedding_mode).lower()
        embedding_lambda = float(self.embedding_lambda)
        if mode == 'gene':
            key = 'gene_enhanced_l2norm'
            emb = self.adata.obsm[key]
        elif mode == 'image':
            key = 'image_enhanced_l2norm'
            emb = self.adata.obsm[key]
        elif mode == 'fused':
            key = 'fused_features_l2norm'
            emb = self.adata.obsm[key]
        elif mode == 'late_fusion':
            # Late fusion keeps the clustering input stable while allowing image contribution tuning.
            key = f'late_fusion_embedding_lambda_{embedding_lambda:g}'
            emb = self.adata.obsm['gene_enhanced_l2norm'] + embedding_lambda * self.adata.obsm['image_enhanced_l2norm']
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            emb = emb / norm
            self.adata.obsm[key] = emb.astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding_mode: {self.embedding_mode}")
        self.adata.obsm['emb'] = emb.astype(np.float32)
        self.adata.uns['embedding_mode'] = mode
        self.adata.uns['embedding_lambda'] = embedding_lambda
        self.adata.uns['emb_key'] = key
        return key

    def process_image_data(self, image):
        target_size = self.input_image_size
        backbone_stride = 32
        if isinstance(target_size, (tuple, list)):
            target_size = int(np.mean(target_size))
        if target_size % backbone_stride != 0:
            target_size = int(round(target_size / backbone_stride) * backbone_stride)
        self.input_image_size = int(target_size)
        
        # Normalize all supported image inputs to a batched ImageNet-style tensor.
        if isinstance(image, str):
            transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_pil = Image.open(image).convert('RGB')
            image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image_array = image.astype(np.float32) / 255.0
            else:
                image_array = image.astype(np.float32)
            if len(image_array.shape) == 3:
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            elif len(image_array.shape) == 2:
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Unsupported image shape: {image_array.shape}")
            resize_transform = transforms.Resize((target_size, target_size))
            image_tensor = resize_transform(image_tensor)
            normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = normalize_transform(image_tensor)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
        elif torch.is_tensor(image):
            image_tensor = image.to(self.device)
            if image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
            if image_tensor.dim() == 3:
                if image_tensor.shape[0] not in (1,3) and image_tensor.shape[-1] in (1,3):
                    image_tensor = image_tensor.permute(2,0,1)
                if image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.repeat(3,1,1)
                image_tensor = transforms.Resize((target_size, target_size))(image_tensor)
                image_tensor = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(image_tensor)
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.dim() == 4:
                if image_tensor.shape[1] == 1:
                    image_tensor = image_tensor.repeat(1,3,1,1)
                image_tensor = transforms.Resize((target_size, target_size))(image_tensor)
                image_tensor = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(image_tensor)
            else:
                raise ValueError(f"Unsupported image tensor dimensions: {tuple(image_tensor.shape)}")
        else:
            raise TypeError(f"Unsupported image data type: {type(image)}")

        H, W = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        self.image_tensor = image_tensor
        self.image_size = (int(W), int(H))
        return image_tensor

    def _pretrain_loss_dict(self, recon_loss):
        zero = torch.tensor(0.0, device=self.device)
        return {
            'total': recon_loss,
            'recon': recon_loss,
            'contrastive': zero,
            'contrastive_raw': zero,
            'contrastive_weight': zero,
            'contrastive_T': zero,
            's_pos_mean': zero,
            's_neg_topk_mean': zero,
            'consistency': zero,
            'prototype': zero,
            'weight_recon': torch.tensor(float(self.alpha), device=self.device),
            'weight_con': torch.tensor(float(self.beta), device=self.device),
            'weight_cons': torch.tensor(float(self.gamma), device=self.device),
            'weight_proto': torch.tensor(float(self.delta), device=self.device),
        }

    def _run_pretrain_step(self, edge_index, edge_weight):
        _, gene_recon, _, _, _, _, _, _, _ = self.multimodal_model(
            self.features, None, edge_index, edge_weight=edge_weight, pre_train=True
        )
        loss = self.loss_fn.reconstruction_loss(gene_recon, self.features)
        return loss, self._pretrain_loss_dict(loss)

    def _run_joint_step(self, image_data, edge_index, edge_weight):
        _, gene_enhanced, image_enhanced, _, fused_gene, fused_image, _, _ = \
            self.multimodal_model(self.features, image_data, edge_index, edge_weight=edge_weight)

        _, gene_recon = self.multimodal_model.gene_encoder(
            self.features, edge_index, edge_weight=edge_weight, pre_train=True
        )

        pos_mask = (self.graph_neigh > 0).to(torch.bool).to(self.device)
        con_out = self.multimodal_model.compute_cross_modal_loss(
            gene_enhanced, image_enhanced, pos_mask=pos_mask
        )

        base_losses = self.loss_fn(
            gene_recon=gene_recon,
            gene_target=self.features,
            gene_features=gene_enhanced,
            image_features=image_enhanced,
            fused_gene=fused_gene,
            fused_image=fused_image,
        )

        total_loss = (
            self.loss_fn.alpha * base_losses['recon'] +
            self.loss_fn.beta  * con_out['loss'] +
            self.loss_fn.gamma * base_losses['consistency'] +
            self.loss_fn.delta * base_losses.get('prototype', torch.tensor(0.0, device=self.device))
        )

        return total_loss, {
            'total': total_loss,
            'recon': base_losses['recon'],
            'contrastive': con_out['loss'],
            'contrastive_raw': torch.as_tensor(con_out['raw_loss'], device=self.device),
            'contrastive_weight': torch.as_tensor(con_out['weight'], device=self.device),
            'contrastive_T': torch.as_tensor(con_out['temperature'], device=self.device),
            's_pos_mean': torch.as_tensor(con_out['s_pos_mean'], device=self.device),
            's_neg_topk_mean': torch.as_tensor(con_out['s_neg_topk_mean'], device=self.device),
            'consistency': base_losses['consistency'],
            'prototype': base_losses.get('prototype', torch.tensor(0.0, device=self.device)),
            'weight_recon': torch.tensor(float(self.loss_fn.alpha), device=self.device),
            'weight_con': torch.tensor(float(self.loss_fn.beta), device=self.device),
            'weight_cons': torch.tensor(float(self.loss_fn.gamma), device=self.device),
            'weight_proto': torch.tensor(float(self.loss_fn.delta), device=self.device),
        }

    def _run_train_step(self, stage, image_data, edge_index, edge_weight):
        if stage == 'pretrain':
            return self._run_pretrain_step(edge_index, edge_weight)
        return self._run_joint_step(image_data, edge_index, edge_weight)

    def _loss_item(self, tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            if tensor_like.numel() == 1 and torch.isfinite(tensor_like):
                return float(tensor_like.detach().item())
            return float('nan')
        try:
            return float(tensor_like)
        except Exception:
            return float('nan')

    def _build_history_record(self, epoch, stage, loss, loss_dict):
        record = {
            'epoch': epoch + 1,
            'stage': stage,
            'loss': float(loss.detach().item()),
            'lr': float(self.optimizer.param_groups[0]['lr']),
        }
        for key, value in loss_dict.items():
            record[key] = self._loss_item(value)
        return record

    def train_multimodal(self):
        morphology_matrix = self._get_morphology_matrix()
        morphology_input_dim = None if morphology_matrix is None else int(morphology_matrix.shape[1])
        model_config = dict(self.model_config)
        if morphology_input_dim is not None:
            model_config['use_precomputed_morphology'] = True
        self.multimodal_model = MultimodalIntegrationModel(
            gene_input_dim=self.dim_input,
            num_cell_types=self.num_cell_types,
            # Keep the historical 512 hidden size to preserve prior experiment results.
            gene_hidden_dim=512,
            gene_output_dim=self.dim_output,
            image_output_dim=self.dim_output,
            cross_attn_heads=8,
            config=model_config,
            morph_input_dim=morphology_input_dim,
        ).to(self.device)

        if (
            hasattr(self.multimodal_model, 'image_encoder')
            and hasattr(self.multimodal_model.image_encoder, 'feature_extractor')
        ):
            self.multimodal_model.image_encoder.feature_extractor.eval()
            for m in self.multimodal_model.image_encoder.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
            for p in self.multimodal_model.image_encoder.feature_extractor.parameters():
                p.requires_grad = False
            if (
                hasattr(self.multimodal_model.image_encoder, 'backbone')
                and hasattr(self.multimodal_model.image_encoder.backbone, 'layer4')
            ):
                for p in self.multimodal_model.image_encoder.backbone.layer4.parameters():
                    p.requires_grad = True
                for m in self.multimodal_model.image_encoder.backbone.layer4.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False
        if hasattr(self.multimodal_model.image_encoder, 'projection'):
            for p in self.multimodal_model.image_encoder.projection.parameters():
                p.requires_grad = False

        self.loss_fn = MultimodalLoss(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            consistency_side="gene",
        )

        parameter_groups = [
            {
                "params": self.multimodal_model.gene_encoder.parameters(),
                "lr": self.lr_gene,
                "weight_decay": self.weight_decay,
            },
            {
                "params": self.multimodal_model.cross_attention.parameters(),
                "lr": self.lr_gene,
                "weight_decay": self.weight_decay,
            },
        ]
        if getattr(self.multimodal_model, 'morph_feature_projector', None) is not None:
            parameter_groups.append({
                "params": self.multimodal_model.morph_feature_projector.parameters(),
                "lr": self.lr_gene,
                "weight_decay": self.weight_decay,
            })
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.warmup_epochs = max(1, self.epochs // 10)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs - self.warmup_epochs),
            eta_min=self.learning_rate * 0.01
        )

        amp_enabled = (self.device.type == 'cuda')
        if amp_enabled and not hasattr(self, 'scaler'):
            self.scaler = GradScaler()

        edge_index, edge_weight = self.edge_index, self.edge_weight
        image_data = self._make_model_image_data()

        training_history = []
        self.training_history = pd.DataFrame()
        pretrain_epochs = int(self.epochs * float(self.pretrain_ratio))
        pretrain_epochs = max(0, min(self.epochs, pretrain_epochs))

        def _run_epoch(epoch, stage):
            self.multimodal_model.train()
            if (
                hasattr(self.multimodal_model, 'image_encoder')
                and hasattr(self.multimodal_model.image_encoder, 'feature_extractor')
            ):
                self.multimodal_model.image_encoder.feature_extractor.eval()
                for m in self.multimodal_model.image_encoder.feature_extractor.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            self.optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with autocast():
                    loss, loss_dict = self._run_train_step(stage, image_data, edge_index, edge_weight)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.multimodal_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self._run_train_step(stage, image_data, edge_index, edge_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multimodal_model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if epoch < self.warmup_epochs:
                lr_scale = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * lr_scale
            else:
                self.scheduler.step()

            return self._build_history_record(epoch, stage, loss, loss_dict)

        if pretrain_epochs > 0:
            for epoch in tqdm(range(pretrain_epochs), desc="Gene pretraining"):
                training_history.append(_run_epoch(epoch, 'pretrain'))

        if pretrain_epochs < self.epochs:
            for epoch in tqdm(range(pretrain_epochs, self.epochs), desc="Joint training"):
                training_history.append(_run_epoch(epoch, 'joint'))

        with torch.no_grad():
            self.multimodal_model.eval()
            gene_features, gene_enhanced, image_enhanced, fused_features, _, _, _, _ = \
                self.multimodal_model(self.features, image_data, edge_index, edge_weight=edge_weight)

            fused_features_normalized = F.normalize(fused_features, p=2, dim=1)
            gene_enhanced_normalized = F.normalize(gene_enhanced, p=2, dim=1)
            image_enhanced_normalized = F.normalize(image_enhanced, p=2, dim=1)

            self.adata.obsm['gene_features'] = gene_features.cpu().numpy()
            self.adata.obsm['gene_enhanced'] = gene_enhanced.cpu().numpy()
            self.adata.obsm['image_enhanced'] = image_enhanced.cpu().numpy()
            self.adata.obsm['fused_features'] = fused_features.cpu().numpy()
            self.adata.obsm['fused_features_l2norm'] = fused_features_normalized.cpu().numpy()
            self.adata.obsm['gene_enhanced_l2norm'] = gene_enhanced_normalized.cpu().numpy()
            self.adata.obsm['image_enhanced_l2norm'] = image_enhanced_normalized.cpu().numpy()

            self._select_embedding_for_clustering()

            self.training_history = pd.DataFrame(training_history)
            return self.adata

    def predict_cell_types(self, adata=None):
        if adata is None:
            adata = self.adata
        if not hasattr(self, 'multimodal_model'):
            raise ValueError("Model not trained. Please run train_multimodal() first.")
        if 'feat' not in adata.obsm:
            get_feature(adata)
        features = torch.FloatTensor(adata.obsm['feat']).to(self.device)
        if 'adj' not in adata.obsm:
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(adata)
            else:
                construct_interaction(adata)
        if self.datatype in ['Stereo', 'Slide']:
            adj = preprocess_adj_sparse(adata.obsm['adj']).to(self.device)
            edge_index = adj.indices().to(self.device)
        else:
            adj = preprocess_adj(adata.obsm['adj'])
            adj = torch.FloatTensor(adj).to(self.device)
            edge_index = adj.nonzero().t().contiguous().to(self.device)
        if self.image_tensor is not None:
            image_size = self.image_tensor.shape[-2:]
            H_img, W_img = int(image_size[0]), int(image_size[1])
            roi_boxes = make_roi_boxes_from_adata(
                adata, patch_size=self.patch_size, image_size=(W_img, H_img)
            )
            image_data = {
                "image": self.image_tensor,
                "roi_boxes": roi_boxes,
                "image_size": (W_img, H_img)
            }
        else:
            image_data = None
        with torch.no_grad():
            self.multimodal_model.eval()
            _, _, _, fused_features, _, _, _, _ = self.multimodal_model(
                features,
                image_data,
                edge_index,
                edge_weight=self.edge_weight,
            )
            cell_proportions, _ = self.multimodal_model.cell_predictor(fused_features)
            adata.obsm['cell_type_proportions'] = cell_proportions.cpu().numpy()
            return adata

    def calculate_metrics(self, pred_col='domain', true_col='ground_truth', mode='auto'):
        adata = self.adata
        metrics_dict = {}
        if pred_col not in adata.obs:
            raise ValueError(f"adata.obs['{pred_col}'] is required to calculate metrics.")
        pred_labels = adata.obs[pred_col]
        has_true = (true_col in adata.obs) and pd.Series(adata.obs[true_col]).notna().any()
        if mode == 'auto':
            eval_mode = 'supervised' if has_true else 'unsupervised'
        else:
            if mode not in ['supervised', 'unsupervised']:
                raise ValueError("mode must be 'auto', 'supervised', or 'unsupervised'")
            eval_mode = mode
        if 'emb_pca' in adata.obsm:
            X_eval = np.asarray(adata.obsm['emb_pca'])
        else:
            X_eval = adata.obsm.get('emb', None)
            if X_eval is not None:
                X_eval = np.asarray(X_eval)
        if has_true:
            true_labels_all = adata.obs[true_col]
            eval_mask = pd.notna(true_labels_all) & pd.notna(pred_labels)
        else:
            true_labels_all = None
            eval_mask = pd.notna(pred_labels)
        if eval_mode == 'supervised':
            if not has_true:
                raise ValueError(f"Supervised metrics require non-empty adata.obs['{true_col}'].")
            true_labels = adata.obs[true_col][eval_mask]
            pred_labels_sup = pred_labels[eval_mask]
            if len(true_labels) > 0:
                ari = metrics.adjusted_rand_score(true_labels, pred_labels_sup)
                nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels_sup)
                metrics_dict.update({'ARI': ari, 'NMI': nmi})
                adata.uns['ari'] = ari
                adata.uns['nmi'] = nmi
        elif eval_mode == 'unsupervised':
            if X_eval is None:
                adata.uns['metrics'] = metrics_dict
                return metrics_dict
            labels = np.asarray(pred_labels[eval_mask])
            X_used = X_eval[np.asarray(eval_mask)]
            n = int(X_used.shape[0])
            k = len(np.unique(labels)) if n > 0 else 0
            valid_k = (2 <= k <= n - 1)
            sc_val = silhouette_score(X_used, labels) if valid_k else float('nan')
            db_val = davies_bouldin_score(X_used, labels) if valid_k else float('nan')
            ch_val = calinski_harabasz_score(X_used, labels) if valid_k else float('nan')

            def _s_dbw_index(X, labels):
                X = np.asarray(X, dtype=float)
                labels = np.asarray(labels)
                uniq = np.unique(labels)
                k = len(uniq)
                n, p = X.shape
                if k < 2 or n <= 1:
                    return np.nan
                clusters = [X[labels == u] for u in uniq]
                counts = np.array([c.shape[0] for c in clusters])
                if np.any(counts < 2):
                    return np.nan
                centers = np.vstack([c.mean(axis=0) for c in clusters])
                cluster_std_vecs = np.vstack([np.std(c, axis=0, ddof=1) for c in clusters])
                global_std_vec = np.std(X, axis=0, ddof=1)
                global_std_norm = np.linalg.norm(global_std_vec)
                if global_std_norm == 0:
                    return np.nan
                scat = np.mean([np.linalg.norm(cluster_std_vecs[i]) / global_std_norm for i in range(k)])
                stdev = np.sqrt(np.sum(np.linalg.norm(cluster_std_vecs, axis=1))) / k
                if stdev <= 0:
                    return scat
                def density(points, center):
                    d = np.linalg.norm(points - center, axis=1)
                    return np.sum(d <= stdev)
                dens_centers = np.array([density(clusters[i], centers[i]) for i in range(k)], dtype=float)
                dens_bw_sum = 0.0
                pair_count = 0
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            continue
                        pair_points = np.vstack([clusters[i], clusters[j]])
                        mid_ij = 0.5 * (centers[i] + centers[j])
                        dens_mid = density(pair_points, mid_ij)
                        denom = max(dens_centers[i], dens_centers[j])
                        if denom > 0:
                            dens_bw_sum += dens_mid / denom
                        else:
                            dens_bw_sum += 0.0
                        pair_count += 1
                dens_bw = dens_bw_sum / pair_count if pair_count > 0 else 0.0
                return float(scat + dens_bw)
            s_dbw_val = _s_dbw_index(X_used, labels) if valid_k else float('nan')
            metrics_dict.update({'SC': sc_val, 'CH': ch_val, 'DB': db_val, 'S_Dbw': s_dbw_val})
            adata.uns['sc'] = sc_val
            adata.uns['db'] = db_val
            adata.uns['ch'] = ch_val
            adata.uns['s_dbw'] = s_dbw_val
        adata.uns['metrics'] = metrics_dict
        return metrics_dict

    def cluster(
        self,
        tool='mclust',
        radius=50,
        key='emb',
        start=0.01,
        end=0.27,
        increment=0.005,
        refinement=True,
        n_clusters=None,
    ):
        if n_clusters is None:
            n_clusters = self.num_cell_types
        has_label = ('ground_truth' in self.adata.obs) and pd.Series(self.adata.obs['ground_truth']).notna().any()
        if (not has_label) and (tool == 'mclust'):
            tool = 'kmeans'
        if tool == 'mclust':
            clustering(
                self.adata,
                n_clusters=n_clusters,
                radius=radius,
                key=key,
                method='mclust',
                refinement=refinement,
            )
        elif tool in ['leiden', 'louvain']:
            clustering(
                self.adata,
                n_clusters=n_clusters,
                radius=radius,
                key=key,
                method=tool,
                start=start,
                end=end,
                increment=increment,
                refinement=True,
            )
        else:
            clustering(self.adata, n_clusters=n_clusters, radius=radius, key=key, method=tool, refinement=refinement)
        eval_mode = 'supervised' if has_label else 'unsupervised'
        _ = self.calculate_metrics(pred_col='domain', true_col='ground_truth', mode=eval_mode)
