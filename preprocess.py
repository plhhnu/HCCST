import random

import numpy as np
import ot
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors


def construct_interaction(adata, n_neighbors=3):
    """Build a symmetric spatial neighbor graph from pairwise Euclidean distances."""
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm['adj'] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    """Build a symmetric spatial neighbor graph using sklearn KNN search."""
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)

    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()

    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm['adj'] = adj
    print('Graph constructed!')


def preprocess(adata):
    """Run the standard Scanpy preprocessing pipeline used by HCCST."""
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def get_feature(adata, deconvolution=False):
    """Store the model feature matrix and its row-permuted augmentation in adata."""
    if deconvolution:
        adata_vars = adata
    else:
        adata_vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_vars.X, csc_matrix) or isinstance(adata_vars.X, csr_matrix):
        feat = adata_vars.X.toarray()[:, ]
    else:
        feat = adata_vars.X[:, ]

    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a


def add_contrastive_label(adata):
    """Add legacy two-column contrastive labels to adata.obsm."""
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def normalize_adj(adj):
    """Apply symmetric adjacency normalization: D^(-1/2) A D^(-1/2)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Normalize an adjacency matrix and add self-loops."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    """Normalize an adjacency matrix with self-loops and return a sparse tensor."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def fix_seed(seed):
    """Set random seeds for reproducible runs."""
    import os

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def permutation(feature, seed=None):
    """Return a row-permuted copy of a feature matrix."""
    if seed is not None:
        current_state = np.random.get_state()
        np.random.seed(seed)

    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    if seed is not None:
        np.random.set_state(current_state)

    return feature_permutated
