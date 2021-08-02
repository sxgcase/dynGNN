import torch
import torch.nn as nn

from model import layers

class STGCN_ChebConv(nn.Module):
    # STGCN(ChebConv) contains 'TGTND TGTND TNFF' structure
    # ChebConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind to
    # approximate graph convolution kernel from Spectral CNN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate):
        super(STGCN_ChebConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], gated_act_func, graph_conv_type, chebconv_matrix, drop_rate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
             self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, gated_act_func, drop_rate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.softsign = nn.Softsign()
            self.relu = nn.ReLU()
            self.softplus = nn.Softplus()
            self.leakyrelu = nn.LeakyReLU()
            self.prelu = nn.PReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'softsign':
                x_act_func = self.softsign(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'softplus':
                x_act_func = self.softplus(x_fc1)
            elif self.act_func == 'leakyrelu':
                x_act_func = self.leakyrelu(x_fc1)
            elif self.act_func == 'prelu':
                x_act_func = self.prelu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            x_out = x_fc2
        return x_out

class STGCN_GCNConv(nn.Module):
    # STGCN(GCNConv) contains 'TGTND TGTND TNFF' structure
    # GCNConv is the graph convolution from GCN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate):
        super(STGCN_GCNConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, gated_act_func, drop_rate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.softsign = nn.Softsign()
            self.relu = nn.ReLU()
            self.softplus = nn.Softplus()
            self.leakyrelu = nn.LeakyReLU()
            self.prelu = nn.PReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'softsign':
                x_act_func = self.softsign(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'softplus':
                x_act_func = self.softplus(x_fc1)
            elif self.act_func == 'leakyrelu':
                x_act_func = self.leakyrelu(x_fc1)
            elif self.act_func == 'prelu':
                x_act_func = self.prelu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            x_out = x_fc2
        return x_out