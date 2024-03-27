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
#        in_edges_layer = th.zeros((len(g.dstnodes()), max_neighs), dtype=th.int64, device=block.device)
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
#        return output
    
    def forward(self,g,h):
        '''Local Attention: Update features layers'''
        feats = self.LocalAttention(g,h)   # [1440, 90]  d=90
        '''Global Attention: Calculate weight layers'''
        weight = self.GlobalAttention(g,h)   # [1440,1]
        
        '''Classifier'''
        updatafeat = weight * feats                                # [1440, 90]
        with g.local_scope(): #临时修改图的特征，不会影响图中的原始特征值
            g.ndata['h'] = updatafeat
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.conv = CNN()
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#        nn.init.xavier_normal_(self.convrow, gain=gain)
#        nn.init.xavier_normal_(self.convcol, gain=gain)
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}
    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}
    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = self.conv(alpha * nodes.mailbox['z'])
#        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
    def forward(self, g,h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class CNN(th.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.convrow = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1,
                        out_channels=1,
                        kernel_size=(2,self.in_dim)),             # [1440,1,10,90]->[1440,1,9,1]
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),
        )
        self.convcol = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1,
                         out_channels=1,
                         kernel_size=(self.max_neighs, 1)),
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),                                 # [1440,1,10,90]->[1440,1,1,90]
        )
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.convrow, gain=gain)
        nn.init.xavier_normal_(self.convcol, gain=gain)

    def forward(self, feats):
        feats = feats.unsqueeze(1)              # [1440,10,90]-> [1440,1,10,90]
        featsrow = self.convrow(feats)          # [1440,1,10,90]->[1440,1,9,1]
        featsrow = np.squeeze(featsrow)         # [1440,1,9,1]->[1440,9]

        featscol = self.convcol(feats)         # [1440,1,10,90]->[1440,1,1,90]
        featscol = np.squeeze(featscol)         # [1440,1,1,90]->[1440,90]
       
        feats = th.cat((featsrow,featscol),dim=1)      # [1440,99]
        return feats
