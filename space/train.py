#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:09:45 2022

@author: liyuzhe
"""
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm.auto import tqdm

from .loss import compute_mmd, mse_loss

        


def adjust_learning_rate(init_lr, optimizer, iteration,seperation):
    lr = max(init_lr * (0.9 ** (iteration//seperation)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        # loss=loss.cpu().detach().numpy()
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss



def train_SPACE_Graph(model, train_data, outdir, epoch=2000,lr=0.005, a=0.2,b=10,loss_type='MSE',patience=50, GPU=0,seed=9):
    
    np.random.seed(seed)
    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(GPU)
        torch.cuda.manual_seed_all(seed)
    else:
        device='cpu'
    
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    epoch_loss = 0.0
    early_stopping = EarlyStopping(patience=patience, checkpoint_file=outdir+'/model.pt')
    
    for epoch in tqdm(range(1, epoch)):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        epoch_lr=adjust_learning_rate(lr, optimizer, epoch, seperation=50)
            
        x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device) 
        optimizer.zero_grad()
            
        z, _ = model.encode(x, edge_index) 
        graph_loss = model.graph_loss(z, train_data.pos_edge_label_index) * a  
        loss = graph_loss  
            
        reconstructed_features = model.decoder_x(z)
        if loss_type=='BCE': 
            feature_loss = torch.nn.functional.binary_cross_entropy(reconstructed_features, x) * b
        elif loss_type=='MSE':
            feature_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * b
            
        loss += feature_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
            
        if epoch%50==0:
            print('====> Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss)) 
            
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print('EarlyStopping: run {} iteration'.format(epoch+1))
            break
    
    return device



def train_SPACE_Gene(model, data, outdir, cluster, epoch, batch_size, lr=0.0005, 
                     weight_decay=5e-4,patience=10,GPU=0, seed=9,beta = 1):
    
    np.random.seed(seed)
    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(GPU)
        torch.cuda.manual_seed_all(seed)
    else:
        device='cpu'
    
    torch.manual_seed(seed)
    model.to(device)
      
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    model.train()
    
    dataset = torch.utils.data.TensorDataset(data,cluster)
    
    early_stopping = EarlyStopping(patience=patience, outdir=outdir+'/model.pt')
    
    for epoch in tqdm(range(1, epoch)):
        epoch_lr = adjust_learning_rate(lr, optimizer, epoch, seperation=10)
        train_data=DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)
        for iteration,data_list in enumerate(train_data):
            x=data_list[0].to(device)
            c=data_list[1].to(device)
            optimizer.zero_grad()
            recon_x = model(x,c)
            mu, log_var = model.encoder(x)     
            z = model.reparameterize(mu, log_var)
            
            true_samples = Variable(torch.randn(x.shape[0], 10), requires_grad=False)
            mmd = compute_mmd(true_samples.to(device), z)
            
            mse=mse_loss(recon_x,x)
            loss=mse+beta*mmd 
            loss.backward()      
            optimizer.step()
        print('====> Epoch: {}, Loss: {:.4f}'.format(epoch+1,loss.cpu().data.numpy()))
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print('EarlyStopping: run {} iteration'.format(epoch))
            break
    
    return device
