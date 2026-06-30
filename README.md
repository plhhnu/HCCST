# HCCST: Histology-Guided Cross-Modal Co-Attention with Multi-Objective Joint Optimization for Spatial Domain Identification

HCCST is a multimodal deep learning framework for spatial domain identification in spatial transcriptomics. It integrates gene expression profiles, spatial coordinates, and histological images to learn robust spot-level representations.

## Framework Overview
![HCCST Framework](Figure/HCCST.png)

The framework mainly includes five components:

- **Morphology-aware graph construction:** HCCST builds a spatial neighborhood graph from spot coordinates and reweights graph edges using ROI-level histological similarity.
- **Gene encoder pretraining:** A GAT-based encoder is pretrained on the morphology-aware graph to learn denoised and structure-aware gene representations.
- **Multimodal representation learning:** Gene embeddings and histological visual tokens are integrated through bidirectional cross-modal co-attention and adaptive gated fusion.
- **Late fusion:** L2-normalized gene and image embeddings are combined to generate unified spot-level representations for downstream analysis.
- **Multi-task optimization:** HCCST is trained with reconstruction, cross-modal contrastive, modality consistency, and prototype regularization losses.

The learned embeddings can be used for spatial clustering, trajectory inference, marker gene visualization, and pseudotime analysis.
## Data availability
All datasets used in this study are publicly available from the original studies or official data repositories. The DLPFC dataset was obtained from the spatialLIBD resource (https://research.libd.org/spatialLIBD/). The Human Breast Cancer (Block A Section 1), Mouse Brain Serial Section 1 (Sagittal-Anterior), Human Breast Cancer (DCIS), and Adult Mouse Brain (FFPE) datasets were obtained from the 10x Genomics public datasets portal: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-0-0, https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-0-0, https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0, and https://www.10xgenomics.com/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0, respectively. The mouse primary visual cortex STARmap dataset and the Human Intestine Cancer (FFPE) dataset used in this study were obtained from the MuCST repository and its associated Zenodo record (https://github.com/xkmaxidian/MuCST; https://doi.org/10.5281/zenodo.10627683). The original Human Intestine Cancer (FFPE) Visium dataset is also publicly available from 10x Genomics (https://www.10xgenomics.com/datasets/human-intestine-cancer-1-standard).

## Installation

#### GPU Acceleration Notice

To fully utilize GPU acceleration, please make sure that the installed PyTorch version is compatible with your CUDA environment. This project has been tested on Linux with Python 3.8.20, an NVIDIA GeForce RTX 3090 GPU, and CUDA Toolkit 11.8. We recommend installing a PyTorch version compatible with CUDA 11.8, such as PyTorch 2.2.2+cu118.

If PyTorch is installed using the default command without specifying the CUDA version, a CPU-only version may be installed, which will prevent the model from using GPU acceleration. Please refer to the official PyTorch installation guide and select the appropriate command according to your CUDA version. For example, for CUDA 11.8, you can use:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

After installation, you can check whether GPU acceleration is available using:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `True`, the GPU environment has been successfully configured.

### Option 1: Install with `environment.yml` Recommended

#### Step 1. Create a new conda environment:

```
conda create -n hccst python=3.8 r-base r-essentials -y
```

Activate the environment:

```
conda activate hccst
```

Install the required R package `mclust`:

```
R
```

In the R console, run:

```
install.packages("mclust")
library(mclust)
q()
```

Then return to the terminal and update the environment using `environment.yml`:

```
conda env update -n hccst -f environment.yml
```

#### Step 2. Final Check

After installation, run the following commands:

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import scanpy as sc; import anndata; import sklearn; print('Basic packages installed successfully')"
python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('PyG packages installed successfully')"
```

If all commands run without errors, the environment is ready.

### Option 2: Install Manually Without `environment.yml`

If the automatic installation using `environment.yml` fails, you can install the environment manually.

#### Step 1: Create a conda environment

```
conda create -n hccst python=3.8
conda activate hccst
```

#### Step 2. Install R and mclust

If the project uses `mclust` for clustering, please install R and the R package `mclust`.

```
conda install r-base r-essentials
```

Start R:

```
R
```

Install `mclust` in the R console:

```
install.packages("mclust")
library(mclust)
q()
```

#### Step 3: Install PyTorch with CUDA 11.8

```
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install PyG

```
pip install torch_geometric==2.5.2
pip install torch_scatter==2.1.2+pt22cu118 torch_sparse==0.6.18+pt22cu118 torch_cluster==1.6.3+pt22cu118 torch_spline_conv==1.2.2+pt22cu118 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

#### Step 5: Install Common Packages

```
pip install scanpy==1.9.3 anndata==0.9.1 numpy==1.22.4 pandas==2.0.3 scipy==1.10.1 scikit-learn==1.3.0 matplotlib==3.7.1 seaborn==0.13.2 tqdm==4.65.0
```

#### Step 6: Install Other Packages

```
pip install squidpy==1.2.3 scikit-image==0.21.0 scikit-misc==0.2.0 rpy2==3.5.11 umap-learn==0.5.7 leidenalg==0.10.2 louvain==0.8.2
```

#### Step 7. Final Check

After installation, run the following commands:

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import scanpy as sc; import anndata; import sklearn; print('Basic packages installed successfully')"
python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('PyG packages installed successfully')"
```

If all commands run without errors, the environment is ready.

## Parameter Settings

HCCST uses a unified training configuration for all datasets, with only a few dataset-specific parameters adjusted according to tissue scale, spot density, and annotation availability.

### Core Parameters

The following parameters are shared across all datasets unless otherwise specified:

| Parameter        | Description                                       | Value         |
| ---------------- | ------------------------------------------------- | ------------- |
| `gamma`          | Weight of the consistency loss                    | 0.1           |
| `delta`          | Weight of the prototype regularization loss       | 0.7           |
| `lr_gene`        | Learning rate for the gene encoder                | 5e-4          |
| `pretrain_ratio` | Ratio of pretraining epochs before joint training | 0.3           |
| `weight_decay`   | Weight decay for regularization                   | 5e-4          |
| `image_size`     | Input image size                                  | 224           |
| `dim_output`     | Dimension of the final embedding                  | 48            |
| `embedding_mode` | Strategy for generating the final embedding       | `late_fusion` |

### Dataset-Specific Parameters

| Dataset                                          | Annotation | Epochs | `embedding_lambda` | `refinement` | `radius` |
| ------------------------------------------------ | ---------- | ------ | ------------------ | ------------ | -------- |
| DLPFC                                            | Labeled    | 550    | 0.1                | `True`       | 12       |
| Mouse Brain Serial Section 1 (Sagittal-Anterior) | Labeled    | 550    | 0.1                | `True`       | 50       |
| Human Breast Cancer (Block A Section 1)          | Labeled    | 550    | 0.1                | `True`       | 12       |
| Mouse Visual Cortex                              | Labeled    | 550    | 0.1                | `True`       | 40       |
| Human Breast Cancer (DCIS)                       | Unlabeled  | 180    | 0.4                | `False`      | 12       |
| Adult Mouse Brain (FFPE)                         | Unlabeled  | 180    | 0.4                | `False`      | 12       |
| Human Intestine Cancer (FFPE)                    | Unlabeled  | 180    | 0.4                | `False`      | 4        |

### Notes

- `radius` controls the neighborhood size used for graph construction and should be adjusted according to the spatial scale and spot density of each dataset.
- `refinement` is enabled for datasets with manual annotations and disabled for datasets without reliable reference boundaries.
- For new datasets, we recommend starting from the default configuration and mainly tuning `radius`, `epochs`, and `embedding_lambda`.
- By default, datasets with manual annotations are clustered using `mclust`, with the number of clusters set to the number of annotated regions. For datasets without manual annotations, `k`-means clustering is used by default, and spatial refinement is disabled to avoid introducing potential bias without reliable reference boundaries.

## Tutorial
You can access the tutorial notebooks for each dataset here:
https://github.com/plhhnu/HCCST/blob/main/Tutorial



### Reference:

Please consider citing the following reference:

```

```

