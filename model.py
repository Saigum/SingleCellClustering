import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch_geometric as tg
from torch_geometric.nn import GATv2Conv,GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
import warnings
import matplotlib.pyplot as plt
from torch import randn_like
import pytorch_lightning as pl
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
    Compute soft assignments for each sample in z to torche centroids.
    Here we use a Student's t–like kernel:
      q_ik = 1 / (1 + ||z_i - μ_k||)
    and torchen normalize over clusters.
    """
    # z: (N, latent_dim); centroids: (n_clusters, latent_dim)
    # z is of dim N x latent_dim
    # centroids is of dim n_clusters x latent_dim
    ## diff is n,1,latent_dim = 1,n_clusters,latent_dim
    ## norm is n,n_clusters
    distances = torch.norm(z[:, None, :] - centroids[None,
                        :, :], dim=2)  # (N, n_clusters)
    q = 1.0 / (1.0 + distances)
    q = q / torch.sum(q, dim=1, keepdim=True)
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

from torch_geometric import nn as gnn

class DualGraphEncoder(nn.Module):
    def __init__(self,
                 n_cells:int,
                 n_genes:int,
                 timepoints:int,
                 ):
        super(DualGraphEncoder,self).__init__()
        ## assuming torchey pass in n_cells,n_genes
        self.cell_graph_encoder= nn.ModuleList(
            [gnn.SAGEConv(in_channels=n_genes,out_channels=n_genes)]*timepoints)
        self.gene_graph_encoder = nn.ModuleList(
            [gnn.SAGEConv(in_channels=n_cells,out_channels=n_cells)]*timepoints)
    def forward(self,x,cell_graph,gene_graph):
        for i in range(len(self.cell_graph_encoder)):
            x = self.cell_graph_encoder[i](x,cell_graph)
        x=x.T
        for i in range(len(self.gene_graph_encoder)):
            x = self.gene_graph_encoder[i](x,gene_graph)
        return x

from torch.nn import functional
class GraphAutoEncoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dims:list,
                 ):
        super(GraphAutoEncoder,self).__init__()
        layers=[input_dim] + hidden_dims
        self.encoder = nn.ModuleList([
            gnn.GATv2Conv(layers[i],layers[i+1])
            for i in range(len(layers)-1)
        ])
        self.decoder = nn.ModuleList([
            gnn.GATv2Conv(layers[i],layers[i+1])
            for i in range(len(layers)-1,-1,-1)
        ])
        self.mu = nn.Linear(hidden_dims[-1],hidden_dims[-1])
        self.logvar = nn.Linear(hidden_dims[-1],hidden_dims[-1])
    def reparametrize(self,mu,logvar):
        sample = mu + 0.5*torch.exp(logvar)*randn_like(logvar)
        return sample
    def forward(self,x,edge_index,edge_feat):
        reduced=x
        for i in range(len(self.encoder)):
            reduced = self.encoder[i](reduced,edge_index,edge_feat)
        mu = self.mu(reduced)
        logvar = self.logvar(reduced)
        z = self.reparametrize(mu,logvar)
        ## !TODO: Write KL Divergence Loss over here.
        reconstructed = z
        for i in range(len(self.decoder)):
            reconstructed = self.decoder[i](reconstructed,edge_index,edge_feat)
        return z,reconstructed

class DualEncoder(nn.Module):
    def __init__(self,
                 n_cells:int,
                 n_genes:int,
                 timepoints:int,
                 hidden_dims:list):
        super(DualEncoder,self).__init__()
        self.dge = DualGraphEncoder(n_cells,n_genes,timepoints)
        self.cell_ae = GraphAutoEncoder(n_genes,hidden_dims)
        self.gene_ae = GraphAutoEncoder(n_cells,hidden_dims)
    def forward(self,
                x,
                cell_edge_index,
                cell_edge_feat,
                gene_edge_index,
                gene_edge_feat):
        impute = self.dge(x,cell_edge_index,gene_edge_index)
        cell_embeddings,cell_reconstructed = self.cell_ae(impute,cell_edge_index,cell_edge_feat)
        gene_embeddings,gene_reconstructed = self.gene_ae(impute,gene_edge_index,gene_edge_feat)
        return cell_embeddings,cell_reconstructed,gene_embeddings,gene_reconstructed
    
import lightning as L



class LightningDualEncoder(L.LightningModule):
    def __init__(self,
                 n_cells:int,
                 n_genes:int,
                 timepoints:int,
                 hidden_dims:list,
                 lr:int=0.001):
        super(LightningDualEncoder,self).__init__()
        self.dge = DualGraphEncoder(n_cells,n_genes,timepoints)
        self.cell_ae = GraphAutoEncoder(n_genes,hidden_dims)
        self.gene_ae = GraphAutoEncoder(n_cells,hidden_dims)
        self.lr = lr

    def forward(self,
                x,
                cell_edge_index,
                cell_edge_feat,
                gene_edge_index,
                gene_edge_feat):
        impute = self.dge(x,cell_edge_index,gene_edge_index)
        cell_embeddings,cell_reconstructed = self.cell_ae(impute,cell_edge_index,cell_edge_feat)
        gene_embeddings,gene_reconstructed = self.gene_ae(impute,gene_edge_index,gene_edge_feat)
        return cell_embeddings,cell_reconstructed,gene_embeddings,gene_reconstructed

    def training_step(self, batch, batch_idx):
        x,cell_edge_index,cell_edge_feat,gene_edge_index,gene_edge_feat = batch
        cell_embeddings,cell_reconstructed,gene_embeddings,gene_reconstructed= self(x,
                cell_edge_index,
                cell_edge_feat,
                gene_edge_index,
                gene_edge_feat
        )

        
        

        loss_dict={}
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    

