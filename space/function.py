#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:21:44 2022

@author: liyuzhe
"""
import os
import random
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sci
import squidpy as sq
import networkx as nx
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data

from .layer import GAT_Encoder
from .model import SPACE_Graph, SPACE_Gene
from .train import train_SPACE_Graph, train_SPACE_Gene
from .utils import graph_construction



def SPACE(adata,k=20,alpha=0.05,seed=42,GPU=0,epoch=2000,lr=0.005,patience=50,outdir='./',loss_type='MSE',verbose=False):
    """
    Single-Cell integrative Analysis via Latent feature Extraction
    
    Parameters
    ----------
    adata
        An AnnData Object with spot/cell x gene matrix stored.
    k
        The number cells consider when constructing Adjacency Matrix. Default: 20.
    alpha
        The relative ratio of reconstruction loss of Adjacency Matrix. Default: 0.05 (0.5 was recommanded for Visium data's domain finding).
    lr
        Learning rate. Default: 2e-4.
    patience
        Patience in early stopping. Default: 50.
    epoch
        Max iterations for training. Default: 2000.
    loss_type
        Loss Function of feature matrix reconstruction loss. Default: MSE (BCE was recommanded for Visium data's domain finding). 
    GPU
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: '.'.
    verbose
        Verbosity, True or False. Default: False.
    
    
    Returns
    -------
    adata with the low-dimensional representation of the data stored at adata.obsm['latent'], and calculated neighbors and Umap. 
    The output folder contains:
    adata.h5ad
        The AnnData Object with the low-dimensional representation of the data stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model.
    """
    
    np.random.seed(seed) 
    torch.manual_seed(seed)
    random.seed(seed)
    
    os.makedirs(outdir, exist_ok=True)
        
    print('Construct Graph')
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0],k=k)
    adj=graph_dict.toarray()
    print('Average links: {:.2f}'.format(np.sum(adj>0)/adj.shape[0]))

    G = nx.from_numpy_array(adj).to_undirected() 
    edge_index = (torch_geometric.utils.convert.from_networkx(G)).edge_index
    
    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()
        
    if loss_type == 'BCE':
        scaler = MaxAbsScaler() # MinMaxScaler() 
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    elif loss_type == 'MSE':
        scaled_x = torch.from_numpy(X_hvg) 
    
    #prepare training data
    data_obj = Data(edge_index=edge_index, x=scaled_x) 
    data_obj.num_nodes = X_hvg.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, val_data, test_data = transform(data_obj) 
    num_features = data_obj.num_features
    
    print('Load SPACE Graph model')
    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first':6,'second':6,'mean':6,'std':6},
        hidden_dims=[128,128],
        dropout=[0.3,0.3,0.3,0.3],
        concat={'first': True, 'second': True})
    model = SPACE_Graph(encoder= encoder,decoder=None,loss_type=loss_type)
    print(model) 
    print('Train SPACE Graph model')
    
    device=train_SPACE_Graph(model, train_data, epoch=epoch,lr=lr, patience=patience,GPU=GPU, seed=seed,a=alpha,loss_type=loss_type,
                                 outdir= outdir)

    pretrained_dict = torch.load(outdir+'/model.pt', map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    model = model.eval()
    node_embeddings = []

    x, edge_index = data_obj.x.to(torch.float).to(device), data_obj.edge_index.to(torch.long).to(device)
    z_nodes, attn_w = model.encode(x, edge_index)
    node_embeddings.append(z_nodes.cpu().detach().numpy()) 
    node_embeddings = np.array(node_embeddings) 
    node_embeddings = node_embeddings.squeeze()

    adata.obsm['latent']=node_embeddings 
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10,use_rep='latent',random_state=seed,key_added='SPACE') 
    sc.tl.umap(adata,random_state=seed, neighbors_key='SPACE')
    
    #Output
    adata.write(outdir+'/adata.h5ad')   
    return adata
    
def SPACE_Cell_Com(adata, cluster, n_neighs=30,resolution=0.3):
    
    sq.gr.spatial_neighbors(adata,n_neighs=n_neighs)
    adj=adata.obsp['spatial_connectivities'].toarray().copy()
    print('Extract Cell Community features')
    df=pd.DataFrame(index=adata.obs.index,columns=adata.obs[cluster].cat.categories)
    for i in range(len(adj)):
        tmp=adata[adj[i].nonzero()]
        df_tmp=tmp.obs[cluster].value_counts(normalize=False)
        df.iloc[i][df_tmp.index]=df_tmp
    df=df.fillna(0)
    df_sum=df.sum(axis=1)
    df=df.div(df_sum,axis=0)
    
    adata_cc=anndata.AnnData(df)
    adata_cc.obs=adata.obs
    adata_cc.obsm['ccom']=adata_cc.X.copy()
    adata_cc.raw=adata_cc
    adata.obsm['ccom']=adata_cc.X.copy()
    adata_cc.obsm['spatial']=adata.obsm['spatial'].copy()
    sc.pp.neighbors(adata_cc, n_neighbors=10, n_pcs=40, use_rep='ccom')
    sc.tl.umap(adata_cc)
    sc.tl.leiden(adata_cc,resolution=resolution)
    
    adata.obs['cell_community']=adata_cc.obs.leiden.copy()
    
    return adata, adata_cc
  

def SPACE_VAE(adata, epoch=1000, batch_size=64, lr=0.0005, weight_decay=5e-4, patience=20, outdir='./',seed=78,GPU=0,beta=1):
    os.makedirs(outdir, exist_ok=True)
    cell_df=adata.obs[['cell_community']]
    cell_df.columns=['cell_community']
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(cell_df[['cell_community']]).toarray())
    cluster = torch.from_numpy(enc_df.values.astype(np.float32))
    x_dim=adata.shape[1]
    c_dim=cluster.shape[1]
    model=SPACE_Gene(x_dim, c_dim)
    data=torch.from_numpy(adata.X.toarray())
    device=train_SPACE_Gene(model, data, outdir=outdir,cluster=cluster,epoch=epoch, batch_size=batch_size, lr=lr, 
                     weight_decay=weight_decay,patience=patience,GPU=GPU, seed=seed,beta = beta)
    
    pretrained_dict = torch.load(outdir+'/model.pt', map_location=device)                            
    model_dict = model.state_dict() 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    model_dict.update(pretrained_dict)  
    model.load_state_dict(model_dict) 
    
    model.eval() 
    data = torch.from_numpy(adata.X.toarray()) 
    latent=model.encodeBatch(data,cluster,out='z') 
    recon_x=model.encodeBatch(data,cluster,out='x') 
    recon_x_=model.encodeBatch(data,cluster,out='x_')
    adata.layers['recon_x']=recon_x.copy()
    adata.layers['recon_x_']=recon_x_.copy()
    adata.obsm['latent']=latent.copy()
    
    adata.write(outdir+'/adata.h5ad')
    
    return adata

    
    
    
    
    
