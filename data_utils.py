from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
import numpy as np
import torch_geometric as tg
import scanpy as sc
import pandas as pd
import anndata as ad
import pickle as pk
from sklearn.preprocessing import LabelEncoder



def normalization(features):
    # features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features


def normalization_for_NE(features):
    # features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 1000000
    features = np.log2(features + 1)
    return features


def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn


def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix


def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W


def getNeMatrix(W_in):
    N = len(W_in)
    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20
    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2
    DD = np.sum(np.abs(W0), axis=0)
    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))
    P = TransitionFields(P, N, eps)
    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T
    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W


def getGraph(features, L, K, method):
    print(f"Constructing graph using {method} method...")
    if method == 'pearson':
        co_matrix = np.corrcoef(features)
    elif method == 'spearman':
        co_matrix, _ = spearmanr(features.T)
    elif method == 'NE':
        co_matrix = np.corrcoef(features)
        features_ne = normalization_for_NE(features)
        in_matrix = np.corrcoef(features_ne)
        NE_matrix = getNeMatrix(in_matrix)
        
        # Apply NE-specific thresholding from scGAC
        N = len(co_matrix)
        sim_sh = 1.
        for i in range(len(NE_matrix)):
            NE_matrix[i][i] = sim_sh * max(NE_matrix[i])
        
        data = NE_matrix.reshape(-1)
        data = np.sort(data)
        data = data[:-int(len(data)*0.02)]
        
        min_sh = data[0]
        max_sh = data[-1]
        delta = (max_sh - min_sh) / 100
    
        temp_cnt = []
        for i in range(20):
            s_sh = min_sh + delta * i
            e_sh = s_sh + delta
            temp_data = data[data > s_sh]
            temp_data = temp_data[temp_data < e_sh]
            temp_cnt.append([(s_sh + e_sh)/2, len(temp_data)])
        
        candi_sh = -1
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1:
                if pear_cnt < temp_cnt[i+1][1] and pear_cnt < temp_cnt[i-1][1]:
                    candi_sh = pear_sh
                    break
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]
                if pear_cnt * 2 < temp_cnt[i-1][1]:
                    candi_sh = pear_sh
        if candi_sh == -1:
            candi_sh = 0.3
        
        propor = len(NE_matrix[NE_matrix <= candi_sh])/(len(NE_matrix)**2)
        propor = 1 - propor
        thres = np.sort(NE_matrix)[:, -int(len(NE_matrix)*propor)]
        co_matrix.T[NE_matrix.T <= thres] = 0
        
    else:
        raise ValueError(f"Unknown graph construction method: {method}")

    N = len(co_matrix)
    up_K = np.sort(co_matrix)[:,-K]

    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    
    thres_L = np.sort(co_matrix.flatten())[-int(((N*N)//(1//(L+1e-8))))]
    mat_K.T[co_matrix.T < thres_L] = 0
    
    # Print statistics
    edges = np.sum(mat_K > 0)
    avg_degree = edges / N
    print(f"Graph statistics:")
    print(f"Nodes: {N}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")
    
    
    return mat_K


def load_data(dataset_name):
    subtype = None
    
    if dataset_name.lower() == 'all':
        # Define the list of available dataset names
        datasets = ['guo', 'biase', 'brown', 'bjorklund', 'chung', 'habib', 'sun', 'pbmc']
        adata_list = []
        # Loop over each dataset name and load the corresponding data
        for ds in datasets:
            adata, subtype = load_data(ds)
            adata_list.append((adata,subtype))
        return adata_list, None

    elif dataset_name.lower() == 'guo':
        adata = sc.read_csv("datasets/Guo/GSE99254.tsv", delimiter="\t").T
        subtype= pd.read_csv("Guo/subtype.ann",delimiter="\t")
        lb = LabelEncoder()
        subtype["label"]=lb.fit_transform(subtype["sampleType"]) 

    elif dataset_name.lower() == "biase":
        adata = sc.read_text("datasets/Biase/GSE57249_fpkm.txt", delimiter="\t")
        subtype = pd.read_csv("datasets/Biase/subtype.ann", delimiter="\t")[["cell", "label"]].values
        celltype_dict = {row[0]: row[1] for row in subtype}
        adata.obs["celltype"] = [celltype_dict.get(name, 'Unknown') for name in adata.obs_names]

    elif dataset_name.lower() == 'brown':
        adata = sc.read_csv("datasets/Brown/hum_melanoma_counts.tsv", delimiter="\t")

    elif dataset_name.lower() == "bjorklund":
        adata = sc.read_csv("datasets/Bjorklund/Bjorklund.tsv", delimiter="\t")
        subtype = pd.read_csv("datasets/Bjorklund/labels.ann", delimiter="\t", header=None).rename(columns={0: "cell", 1: "label"})
        celltype_dict = {}
        for i in range(len(subtype)):
            celltype_dict[subtype.iloc[i]["cell"]] = subtype.iloc[i]["label"]
        adata.obs["celltype"] = [celltype_dict.get(name, 'Unknown') for name in adata.obs_names]
        adata = adata[adata.obs["celltype"] != "Unknown", :].copy()

    elif dataset_name.lower() == 'chung':
        adata = sc.read_csv("datasets/Chung/GSE75688.tsv", delimiter="\t").T
        subtype = pd.read_csv("datasets/Chung/GSE75688_final_sample_information.txt", delimiter="\t")
        subtype = subtype[["sample", "index3"]].values
        celltype_dict = {row[0]: row[1] for row in subtype}
        adata.obs['celltype'] = [celltype_dict.get(name, 'Unknown') for name in adata.obs_names]
        # Filter cells with known celltypes (not 'Unknown')
        known_cells = adata.obs['celltype'] != 'Unknown'
        filtered_adata = adata[known_cells, :]
        # Save expression data
        filtered_adata.to_df().to_csv('datasets/Chung/Chung_data.csv')
        # Save cell type annotations
        labels = pd.DataFrame({
            'cell': filtered_adata.obs.index,
            'label': filtered_adata.obs['celltype']
        })
        labels.to_csv('datasets/Chung/label.ann', sep="\t", index=False)

    elif dataset_name.lower() == 'habib':
        habib = sc.read_csv("datasets/Habib/GSE104525_Mouse_Processed_GTEx_Data.DGE.UMI-Counts.txt", delimiter="\t")
        adata = habib.transpose()

    elif dataset_name.lower() == 'sun':
        adata_list = pk.load(open("datasets/Sun/celltype_specific_counts.pkl", "rb"))
        dfs = []
        for i, adata in enumerate(adata_list):
            temp_df = pd.DataFrame({
                'barcode': adata.obs_names,
                'label': i
            })
            dfs.append(temp_df)
        subtype_df = pd.concat(dfs, ignore_index=True)
        print(subtype_df.head())
        adata = ad.concat(adata_list)
            

    elif dataset_name.lower() == 'pbmc':
        adata = sc.read_10x_mtx("datasets/pbmc/pbmc6k_matrices", var_names='gene_symbols', cache=False)

    elif dataset_name.lower() == "romanov":
        adata = pd.read_excel("/scratch/saigum/codebase/datasets/Romanov/GSE74672_expressed_mols_with_classes.xlsx")
        lb = LabelEncoder()
        subtype = lb.fit_transform(romanov.iloc[0])

    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    
    return adata, subtype


