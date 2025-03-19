import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch_geometric as tg
from torch_geometric.nn import GATv2Conv,GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

##########################################
#         MODEL COMPONENTS
##########################################



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=64,attn_heads=4):
        super(Encoder, self).__init__()
        layers = hidden_dims.copy()
        layers.insert(0, input_dim)
        
        self.layers = nn.ModuleList()
        
        # All layers except last should concat heads
        for i in range(len(layers) - 1):
            out_dim = layers[i + 1] // attn_heads  # Divide by heads since we'll concat
            self.layers.append(
                GATConv(
                    layers[i],
                    out_dim,
                    heads=attn_heads,
                    concat=True,  # Changed to True
                    dropout=0.2,  # Reduced dropout
                    bias=True
                )
            )
            
        # Final layer still averages heads
        self.layers.append(
            GATConv(
                layers[-1],
                output_dim,
                heads=attn_heads,
                concat=False,
                dropout=0.2,
                bias=True
            )
        )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 256, 512], output_dim=2000):
        super(Decoder, self).__init__()
        layers = hidden_dims.copy()
        layers.insert(0, input_dim)
        layers.append(output_dim)
        
        module_list = []
        for i in range(len(layers) - 1):
            module_list.append(nn.Linear(layers[i], layers[i + 1]))
            module_list.append(nn.BatchNorm1d(layers[i + 1]))
            if i < len(layers) - 2:
                module_list.append(nn.ELU())
                module_list.append(nn.Dropout(0.2))  # Reduced dropout
                
        self.network = nn.Sequential(*module_list)

    def forward(self, x):
        return self.network(x)


def softcluster_assignments(z, centroids):
    """
    Compute soft assignments for each sample in z to the centroids.
    Here we use a Student's t–like kernel:
      q_ik = 1 / (1 + ||z_i - μ_k||)
    and then normalize over clusters.
    """
    # z: (N, latent_dim); centroids: (n_clusters, latent_dim)
    # z is of dim N x latent_dim
    # centroids is of dim n_clusters x latent_dim
    ## diff is n,1,latent_dim = 1,n_clusters,latent_dim
    ## norm is n,n_clusters
    distances = th.norm(z[:, None, :] - centroids[None,
                        :, :], dim=2)  # (N, n_clusters)
    q = 1.0 / (1.0 + distances)
    q = q / th.sum(q, dim=1, keepdim=True)
    return q





class scGAC(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, init_centroids=None):
        super(scGAC, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)


    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_recon = self.decoder(z)
        return x_recon, z

##########################################
#         TRAINING FUNCTION
##########################################


def train(model, criterion, optimizer, feautures, edge_index, epochs,lambda_recon=1.0,lambda_kl=0.5):
    """
    Train the model with a combined reconstruction and clustering loss.
    Here we use a KL divergence loss for the clustering (DEC style)
    and a reconstruction loss (e.g., MSE).
    """
    kl_div_loss_fn = nn.KLDivLoss(reduction='batchmean')
    model.train()

    recon_loss_history = []
    kl_loss_history = []
    print(f"Sanity Check: features {feautures.shape} , edge_index {edge_index.shape} ")
    x=feautures
    for epoch in range(epochs):
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        optimizer.zero_grad()
        xhat, z, q, p = model(x, edge_index)
        print(f"Sanity Check: xhat {xhat.shape} , z {z.shape} , q {q.shape} , p {p.shape} ")
        print(q)
        recon_loss = criterion(xhat, x)
        kl_loss = kl_div_loss_fn(
            th.log(q + 1e-10), p.detach())  # kl needs log probs
        loss = lambda_recon*recon_loss + lambda_kl*kl_loss
        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        recon_loss_history.append(total_recon_loss)
        kl_loss_history.append(total_kl_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}: Recon Loss = {total_recon_loss:.4f}, KL Loss = {total_kl_loss:.4f}")

    plt.figure()
    plt.plot(recon_loss_history, label='Reconstruction Loss')
    plt.plot(kl_loss_history, label='KL Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.show()

    return model
