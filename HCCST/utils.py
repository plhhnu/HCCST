import numpy as np
import ot
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Cluster embeddings with R mclust and store labels in adata.obs['mclust']."""
    np.random.seed(random_seed)

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri as numpy2ri

    robjects.r.library("mclust")
    numpy2ri.activate()

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust',
               start=0.1, end=3.0, increment=0.01, refinement=False):
    """Run PCA followed by the selected clustering method."""
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'kmeans':
        X = adata.obsm['emb_pca']
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
        labels = kmeans.fit_predict(X)
        adata.obs['kmeans'] = pd.Categorical(labels.astype(str))
        adata.obs['domain'] = adata.obs['kmeans']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        print(f'Refining labels with radius {radius}')
        use_radius = int(radius)
        new_type = refine_label(adata, use_radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    """Refine labels by assigning each spot the majority label among nearby spots."""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb_pca',
               start=0.1, end=3.0, increment=0.01):
    """Search for a graph-clustering resolution that yields n_clusters labels."""
    print('Searching resolution...')
    label = 0

    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, clusters={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, clusters={}'.format(res, count_unique))

        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "No matching resolution found. Try a wider range or a smaller step."
    return res
