#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:52:20 2022

@author: liyuzhe
"""
import scanpy as sc
import numpy as np
from scipy.spatial import distance
import networkx as nx


# Construct adj graph of image-based data 
def graph_computing(adj_coo, cell_num,k):
    edgeList = []
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, 'euclidean')
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0, res[0][1:k+1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k+1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList


def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def graph_construction(adj_coo, cell_N, k):
    adata_Adj = graph_computing(adj_coo, cell_N, k)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    
    return adj_org


def getNClusters(adata,n_cluster,range_min=0,range_max=3,max_steps=20, method='louvain', key_added=None,neighbors_key=None):
    """
    Function will test different settings of louvain to obtain the target number of clusters.
    adapted from the function from the Pinello lab. See: https://github.com/pinellolab/scATAC-benchmarking
    It can get cluster for both louvain and leiden.
    You can specify the obs variable name as key_added. 
    """
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        print('step ' + str(this_step))
        this_resolution = this_min + ((this_max-this_min)/2)
        
        if (method == 'louvain') and (key_added==None):
            sc.tl.louvain(adata, resolution=this_resolution,neighbors_key=neighbors_key)
        elif method == 'louvain'and isinstance(key_added, str):
            sc.tl.louvain(adata, resolution=this_resolution, key_added=key_added,neighbors_key=neighbors_key)
        elif( method == 'leiden') and (key_added==None):
            sc.tl.leiden(adata,resolution=this_resolution,neighbors_key=neighbors_key)
        else:
            sc.tl.leiden(adata,resolution=this_resolution, key_added=key_added,neighbors_key=neighbors_key)
    
        
        if key_added==None:
            this_clusters = adata.obs[method].nunique()
        else:
            this_clusters = adata.obs[key_added].nunique()
        
        print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))
        
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        elif this_clusters == n_cluster:
            break
            return this_resolution
        else:
            print('Cannot find the number of clusters')
            print('Clustering solution from last iteration is used:' + str(this_clusters) + ' at resolution ' + str(this_resolution))
   
        this_step += 1
        
        
        
        
