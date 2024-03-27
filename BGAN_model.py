"""
Author: Zhiheng Zhou

This module defines a Functional connectom（FC）based（Graph Neural Network (GNN) model for fMRI analysis using PyTorch and dgl.

"""

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import random
from torch.nn.parameter import Parameter
device=th.device('cuda' if th.cuda.is_available() else 'cpu')

class BGAN(nn.Module):
    def __init__(self,in_dim, out_dim, n_classes):
        super(BGAN, self).__init__()
        self.in_dim = in_dim
        self.classify = nn.Linear(out_dim, n_classes)
        self.GATLayer = GATLayer(in_dim,out_dim)
        self.conv = dglnn.GraphConv(in_dim, 1)
        self.localw= Parameter(th.randn(self.max_neighs+self.in_dim-1,out_dim))
        self.ReLU = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=1)
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.localw, gain=gain)

    def get_graphs(self, block):
        g = block
        self.max_neighs = 10

        in_edges = th.tensor([]).to(device)
        out_edges = th.tensor([]).to(device)
        for node in range(len(g.dstnodes())): 
            src, dst = g.in_edges(int(node))
            if src.shape[0] == 0:
                src = th.tensor([int(node)], dtype=th.int64)
            src = src.repeat(self.max_neighs)[:self.max_neighs]
            dst = dst.repeat(self.max_neighs)[:self.max_neighs]
            out_edges = th.cat([out_edges,src]).to(device)
            in_edges = th.cat([in_edges,dst]).to(device)
        graph = dgl.graph((out_edges.long(),in_edges.long())).to(device)
        return graph

    def GlobalAttention(self,g,h):
        globalweight = self.Softmax(self.conv(g,h).reshape(self.conv(g,h).shape[0]//self.in_dim,self.in_dim))
        globalweight = globalweight.reshape(globalweight.shape[0]*self.in_dim,1)
        return globalweight

    def LocalAttention(self,g,h):
        graph = self.get_graphs(g)
        output = self.GATLayer(graph,h)
        updatefeat = self.ReLU(th.mm(output,self.localw) + h)
        return updatefeat
    
    def forward(self,g,h):
        '''Local Attention: Update features layers'''
        feats = self.LocalAttention(g,h)   
        '''Global Attention: Calculate weight layers'''
        weight = self.GlobalAttention(g,h)  
        
        '''Classifier'''
        updatafeat = weight * feats                       
        with g.local_scope(): 
            g.ndata['h'] = updatafeat
            hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.conv = CNN()
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = self.conv(alpha * nodes.mailbox['z'])
        return {'h': h}
    def forward(self, g,h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class CNN(th.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.convrow = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1,
                        out_channels=1,
                        kernel_size=(2,self.in_dim)),           
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),
        )
        self.convcol = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1,
                         out_channels=1,
                         kernel_size=(self.max_neighs, 1)),
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),                                
        )
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.convrow, gain=gain)
        nn.init.xavier_normal_(self.convcol, gain=gain)

    def forward(self, feats):
        feats = feats.unsqueeze(1)              
        featsrow = self.convrow(feats)          
        featsrow = np.squeeze(featsrow)      

        featscol = self.convcol(feats)        
        featscol = np.squeeze(featscol)      
       
        feats = th.cat((featsrow,featscol),dim=1)    
        return feats
