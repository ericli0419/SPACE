#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:33:19 2022

@author: liyuzhe
"""
import torch
import torch.nn as nn
from torch.nn import init, Sequential, Linear, ReLU, BatchNorm1d, Dropout, Sigmoid, LeakyReLU
from torch_geometric.nn import GAE, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from .layer import GAT_Encoder


EPS = 1e-15


class SPACE_Graph(GAE):
    def __init__(self, encoder, decoder,normalize=True):
        super(SPACE_Graph, self).__init__(encoder, decoder)
        
        self.decoder = InnerProductDecoder()
        self.reset_parameters()
        
        if normalize:
            self.decoder_x = Sequential(Linear(in_features=self.encoder.latent_dim, out_features=128),
                                      BatchNorm1d(128),
                                      LeakyReLU(),
                                      Dropout(0.1),
                                      Linear(in_features=128, out_features=self.encoder.in_channels),
                                      Sigmoid())
        else:
            self.decoder_x = Sequential(Linear(in_features=self.encoder.latent_dim, out_features=128),
                                      BatchNorm1d(128),
                                      LeakyReLU(),
                                      Dropout(0.1),
                                      Linear(in_features=128, out_features=self.encoder.in_channels),
                                      ReLU())

    
    def encode(self, *args, **kwargs):
        
        z, attn_w = self.encoder(*args, **kwargs)
        
        return z, attn_w


    
    def graph_loss(self, z, pos_edge_index, neg_edge_index=None):

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    
     
    def load_model(self, path):

        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()




class SPACE_Gene(nn.Module):
    def __init__(self, x_dim, c_dim):
        super(SPACE_Gene, self).__init__() 
        
        # encoder layer
        self.e_fc1 = self.fc_layer(x_dim, 1024,activation=1,dropout=True, dropout_p=0.1)
        self.e_fc2 = self.fc_layer(1024, 128,activation=1,dropout=True, dropout_p=0.1)
        self.mu_enc = nn.Linear(128, 10)
        self.var_enc = nn.Linear(128, 10)

        
        # decoder layer
        self.d_fc1 = self.fc_layer(10, 128,activation=1,dropout=True, dropout_p=0.1)
        self.d_fc2 = self.fc_layer(c_dim, 128,activation=1,dropout=True, dropout_p=0.1)
        self.d_fc3 = self.fc_layer(128, 1024,activation=1,dropout=True, dropout_p=0.1)
        self.d_fc4 = self.fc_layer(1024, x_dim, activation=1)
        

        self.reset_parameters()
    

    def reparameterize(self, mu, log_var):
        # vae reparameterization trick
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        
        self.z_mean = mu
        self.z_sigma = std
        
        return mu + eps*std
        
    
    def forward(self,x,c):
        
        mu, log_var = self.encoder(x)     
        z = self.reparameterize(mu, log_var)
        return self.decoder(z,c)
    
    
    def encoder(self, x): 
        
        layer1 = self.e_fc1(x) 
        layer2 = self.e_fc2(layer1)
        mu=self.mu_enc(layer2)
        log_var=self.var_enc(layer2)

        return mu, log_var
      
    
    def decoder(self, z,c):
        
        recon_x = self.d_fc4(self.d_fc3(self.d_fc1(z)+self.d_fc2(c)))
        return recon_x
 
    
    def fc_layer(self, in_dim, out_dim, activation=0, dropout=False, dropout_p=0.1):
        if dropout:
            if activation == 0:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            elif activation == 1:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout_p))
            elif activation == 2:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
        else:
            if activation == 0:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
            elif activation == 1:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU())
        return layer

    
    
    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    
    def encodeBatch(self, data_x, data_c,device='cuda', out='z'):
        x = data_x.float().to(device)
        c = data_c.float().to(device)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        if out == 'z':
            output=z.detach().cpu().data.numpy()
        elif out == 'x':
            recon_x = self.decoder(z,c)
            output=recon_x.detach().cpu().data.numpy()
        elif out == 'x_':
            recon_x_ = self.d_fc4(self.d_fc3(self.d_fc1(z)))
            output = recon_x_.detach().cpu().data.numpy()
        return output
    
    
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        

    
