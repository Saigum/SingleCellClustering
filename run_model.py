import argparse
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from model import scGAC, train
from data_utils import normalization, getGraph, load_data
import torch as th
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from sklearn.cluster import KMeans
import torch.nn.functional as F
import scanpy as sc
import scipy.sparse
import matplotlib.pyplot as plt


def saveClusterResult(pred_labels, cell_names, dataset_name):
    """Save clustering results to a file."""
    results_df = pd.DataFrame({
        'cell': cell_names,
        'predicted_label': pred_labels
    })
    results_df.to_csv(f"{dataset_name}_predicted_labels.ann", sep="\t", index=False)
    print(f"Saved clustering results to {dataset_name}_predicted_labels.ann")


def target_distribution(q):
    """
    Compute the target distribution p_ij = (q_ij^2 / f_j) / (sum_k(q_ij^2 / f_k))
    where f_j = sum_i(q_ij), as in the DEC paper.
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def soft_clustering(z,n_clusters):
    ## use kmeans for initial centroids
    kmeans = KMeans(n_clusters,n_init=20)
    y_pred = kmeans.fit_predict(z)
    centroids = kmeans.cluster_centers_
    ## calculate the distance matrix
    dist = np.linalg.norm(z[:, np.newaxis] - centroids, axis=2)
    q = 1.0 / (1.0 + dist ** 2)
    q = q ** ((1.0 + 2.0) / 2.0)
    q = (q.T / np.sum(q, axis=1)).T
    return q


# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims, latent_dim):
#         super(Encoder, self).__init__()
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU()
#             ])
#             prev_dim = hidden_dim
            
#         self.encoder = nn.Sequential(*layers)
#         self.latent = nn.Linear(prev_dim, latent_dim)
        
#     def forward(self, x):
#         x = self.encoder(x)
#         z = self.latent(x)
#         return z

# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dims, output_dim):
#         super(Decoder, self).__init__()
#         layers = []
#         prev_dim = latent_dim
        
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU()
#             ])
#             prev_dim = hidden_dim
            
#         layers.append(nn.Linear(prev_dim, output_dim))
#         self.decoder = nn.Sequential(*layers)
        
#     def forward(self, z):
#         return self.decoder(z)

# class SimpleAutoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
#         super(SimpleAutoencoder, self).__init__()
#         self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
#         self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        
#     def forward(self, x):
#         z = self.encoder(x)
#         x_recon = self.decoder(z)
#         return x_recon, z

def train_and_evaluate(adata,subtype,args):
    print(f"Adata for {args.dataset}: {adata}")
    print(f"Dtype of adata: {adata.X.dtype}")
    
    sc.pp.normalize_total(adata, inplace=True,target_sum=1e5)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var['highly_variable']]

    # # Now do PCA on the highly variable genes
    # sc.pp.pca(adata, n_comps=min(512, adata.shape[1]))
    # features = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float)

    # Scale the PCA features
    #features = (features - features.mean(0)) / features.std(0)
    features = torch.tensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X, dtype=torch.float)
    ##cell wise normalize the data
    # features = normalization(features)
    # features = torch.tensor(features, dtype=torch.float)
    print(f"Shape of features: {features.shape}")
    
    # Print Adata statistics
    
    print("\nAdata Statistics:")
    print(f"Number of cells: {features.shape[0]}")
    print(f"Number of genes: {features.shape[1]}")
    print(f"Mean expression: {features.mean():.4f}")
    print(f"Std deviation: {features.std():.4f}")
    print(f"Min value: {features.min():.4f}")
    print(f"Max value: {features.max():.4f}")
    print(f"Sparsity: {(features == 0).sum() / features.numel():.2%}")
    print("--------------------------------")
    print("Building graph using method:", args.graph_method)
    N = len(adata.X)
    avg_N = N // args.clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)
    L = 0

    graph_mat = getGraph(features, L=L, K=K, method=args.graph_method)
    row, col = np.nonzero(graph_mat)
    edge_index = np.vstack((row, col))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    print("\nPrinting Graph statistics:")
    print(f"Number of cells: {N}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Average degree: {edge_index.shape[1] / N:.2f}")
    print(f"Sparsity: {(graph_mat == 0).sum() / graph_mat.size:.2%}")
    print(f"Variance of node degrees: {np.var(np.sum(graph_mat, axis=1)):.2f}")
    print("--------------------------------")

    # if(adata.X.shape[1] < 2048):
    #     print("Not enough highly variable genes, using all genes")
    #     features = torch.tensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X, dtype=torch.float)
    # else:
    #     print("Using PCA with 512 components")
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=2048)
    #     features = th.tensor(pca.fit_transform(adata.X), dtype=torch.float)
    #     #features = torch.tensor(features, dtype=torch.float)
#   Initialize Model
    input_dim = features.shape[1]
    hidden_dims = [int(x.strip()) for x in args.hidden.split(',')]
    print(f"Hidden Dimensions: {hidden_dims}")
    print("Initializing scGAC model...")
    # model = SimpleAutoencoder(input_dim=input_dim, 
    #                         hidden_dims=hidden_dims,
    #                         latent_dim=args.latent_dim, 
    #                         n_clusters=args.clusters)
    model= scGAC(input_dim, hidden_dims, args.latent_dim, args.clusters)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print("Starting training for {} epochs...".format(args.epochs))
    print(
        f"Sanity Check: features {features.shape} , edge_index {edge_index.shape} ")


    print("Feature Statistics:")
    print(f"Number of cells: {features.shape[0]}")
    print(f"Number of genes: {features.shape[1]}")
    print(f"Mean expression: {features.mean():.4f}")
    print(f"Std deviation: {features.std():.4f}")
    print(f"Min value: {features.min():.4f}")
    print(f"Max value: {features.max():.4f}")
    print(f"Sparsity: {(features == 0).sum() / features.numel():.2%}")
    print("--------------------------------")
    
    # Pre-training phase
    print("Starting pre-training phase...")
    model.train()
    pre_optimizer = optim.Adam(model.parameters(), lr=0.001) 

    pretraining_losses = []
    min_loss = float('inf')
    patience = 5
    no_improve = 0
    
    results=[]
    for epoch in range(args.pre_epochs):
        pre_optimizer.zero_grad()
        recon, z = model(features, edge_index)
        loss = F.mse_loss(recon, features)
        loss.backward()
        pre_optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                relative_error = torch.norm(recon - features) / torch.norm(features)
                print(f"Epoch {epoch}:")
                print(f"Loss = {loss.item():.4f}")
                print(f"Relative Error = {relative_error:.4f}")
        
        pretraining_losses.append(loss.item())
    plt.plot(pretraining_losses)
    plt.savefig("../plots/pretraining_losses.png")
    plt.close()
    print("Starting deep clustering phase...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_losses = []
    recon_losses = []
    cluster_losses = []
    ari_scores = []
    nmi_scores = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        recon, z = model(features, edge_index)
        # recon,z = model(features)
        q = soft_clustering(z.detach().numpy(),args.clusters)
        q = torch.tensor(q, dtype=torch.float)
        
        # Update target distribution
        target = target_distribution(q.detach())
        
        # Calculate losses
        recon_loss = F.mse_loss(recon, features)
        cluster_loss = F.kl_div(q.log(), target)
        total_loss = args.c1 * recon_loss + args.c2 * cluster_loss
        total_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        cluster_losses.append(cluster_loss.item())
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Evaluate clustering
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                xhat,z = model(features, edge_index)
                # xhat,z = model(features)
                q = soft_clustering(z.detach().numpy(),args.clusters)
                q = torch.tensor(q, dtype=torch.float)
                y_pred = torch.argmax(q, dim=1).cpu().numpy()
                if subtype is not None:
                    true_labels = subtype['label'].values
                    ari = adjusted_rand_score(true_labels, y_pred)
                    print(f"Epoch {epoch}: ARI = {ari:.4f}, Loss = {total_loss.item():.4f}")
                    ari_scores.append(ari)
                    results.append(y_pred)
                    nmi_scores.append(normalized_mutual_info_score(true_labels, y_pred))
                    
    plt.plot(range(len(ari_scores)),ari_scores)
    plt.savefig("../plots/ari_scores.png")
    plt.close()
    plt.plot(range(len(nmi_scores)),nmi_scores)
    plt.savefig("../plots/nmi_scores.png")
    plt.close()
    print(f"Best ARI: {max(ari_scores)}, Best NMI: {max(nmi_scores)}")
    print("Saving Best Clustering Results...")
    best_idx = np.argmax(ari_scores)
    saveClusterResult(results[best_idx], adata.obs_names, args.dataset)
    
    # Save final results
    plt.plot(range(len(total_losses)),total_losses)
    plt.savefig("../plots/total_losses.png")
    plt.close()
    plt.plot(range(len(recon_losses)),recon_losses)
    plt.savefig("../plots/recon_losses.png")
    plt.close()
    plt.plot(range(len(cluster_losses)),cluster_losses)
    plt.savefig("../plots/cluster_losses.png")
    plt.close()
    model.eval()
    with torch.no_grad():
        # _, _, _, q = model(features, edge_index)
        xhat,z = model(features, edge_index)
        #xhat,z = model(features)
        q = soft_clustering(z.detach().numpy(),args.clusters)
        q = torch.tensor(q, dtype=torch.float)
        final_pred = torch.argmax(q, dim=1).cpu().numpy()
    
    
    saveClusterResult(final_pred, adata.obs_names, f"Final_{args.dataset}")


def main(args):

    # Load Data
    print("Loading dataset from:", args.dataset)
    adata,  = load_data(args.dataset)
    if(args.dataset == "all"):
        datasets = ['guo', 'biase', 'brown', 'bjorklund', 'chung', 'habib', 'sun', 'pbmc']
        for i,adatas,subtype in enumerate(adata):
            train_and_evaluate(adatas,args) 
    else:
        train_and_evaluate(adata,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the scGAC model on a graph dataset.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Path to the dataset (.npy file) containing the feature matrix.")
    parser.add_argument('--clusters', type=int, default=10,
                        help="Number of clusters for clustering layer.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate for optimizer.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument('--latent_dim', type=int, default=128,
                        help="Dimension of the latent representation (encoder output).")
    parser.add_argument('--hidden', type=str, default="512,256,128",
                        help="Comma-separated list of hidden dimensions for the encoder.")
    parser.add_argument('--graph_method', type=str, default="pearson", choices=["pearson", "spearman", "NE"],
                        help="Method to compute the graph connectivity from features.")
    parser.add_argument('--pre_lr', type=float, default=0.001,
                        help="Learning rate for pre-training optimizer.")
    parser.add_argument('--pre_epochs', type=int, default=50,
                        help="Number of pre-training epochs.")
    parser.add_argument('--c1', type=float, default=1.0,
                        help="Weight for reconstruction loss.")
    parser.add_argument('--c2', type=float, default=0.5,
                        help="Weight for cluster loss.")
    parser.add_argument('--subtype_path', type=str, default=None,
                        help="Path to the subtype file.")

    args = parser.parse_args()
    main(args)
