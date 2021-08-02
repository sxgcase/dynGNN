import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

##Before the Temporal Convolution Layer, we need to align the c_in and c_out
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        #H_in and H_out remain the same since padding =0, stide =1
        #W_in and W_out remain the same since padding =0, stide =1
        self.align_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, c_in, timestep, n_vertex = x.shape
            x_align = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x_align = x
        return x_align

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        return result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result

class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |-------------------------------| * residual connection *
    #        |                               |
    #        |    |--->--- casual conv ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- casual conv --- sigmoid ---|                               
    #
    
    #param x: tensor, [batch_size, c_in, timestep, n_vertex]

    def   __init__(self, Kt, c_in, c_out, n_vertex, act_func, enable_gated_act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.act_func = act_func
        self.enable_gated_act_func = enable_gated_act_func
        self.align = Align(self.c_in, self.c_out)
        if self.enable_gated_act_func == True:
            self.causal_conv = CausalConv2d(in_channels=self.c_in, out_channels=2 * self.c_out, kernel_size=(self.Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=(self.Kt, 1), enable_padding=False, dilation=1)
        self.linear = nn.Linear(self.n_vertex, self.n_vertex)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.enable_gated_act_func == True:
            #in the second dimension, split as half
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            # Temporal Convolution Layer (GLU)
            if self.act_func == "glu":

                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x_glu = torch.mul((x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_glu

            # Temporal Convolution Layer (GTU)
            elif self.act_func == "gtu":
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x_gtu = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_gtu

            else:
                raise ValueError(f'ERROR: activation function {self.act_func} is not defined.')

        else:

            # Temporal Convolution Layer (Linear)
            if self.act_func == "linear":
                x_linear = self.linear(x_causal_conv + x_in)
                x_tc_out = x_linear
            
            # Temporal Convolution Layer (Sigmoid)
            elif self.act_func == "sigmoid":
                x_sigmoid = self.sigmoid(x_causal_conv + x_in)
                x_tc_out = x_sigmoid

            # Temporal Convolution Layer (Tanh)
            elif self.act_func == "tanh":
                x_tanh = self.tanh(x_causal_conv + x_in)
                x_tc_out = x_tanh

            # Temporal Convolution Layer (Softsign)
            elif self.act_func == "softsign":
                x_softsign = self.softsign(x_causal_conv + x_in)
                x_tc_out = x_softsign

            # Temporal Convolution Layer (ReLU)
            elif self.act_func == "relu":
                x_relu = self.relu(x_causal_conv + x_in)
                x_tc_out = x_relu

            # Temporal Convolution Layer (Softplus)
            elif self.act_func == "softplus":
                x_softplus = self.softplus(x_causal_conv + x_in)
                x_tc_out = x_softplus
        
            # Temporal Convolution Layer (LeakyReLU)
            elif self.act_func == "leakyrelu":
                x_leakyrelu = self.leakyrelu(x_causal_conv + x_in)
                x_tc_out = x_leakyrelu

            # Temporal Convolution Layer (PReLU)
            elif self.act_func == "prelu":
                x_prelu = self.prelu(x_causal_conv + x_in)
                x_tc_out = x_prelu

            # Temporal Convolution Layer (ELU)
            elif self.act_func == "elu":
                x_elu = self.elu(x_causal_conv + x_in)
                x_tc_out = x_elu

            else:
                raise ValueError(f'ERROR: activation function {self.act_func} is not defined.')
        
        return x_tc_out


class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, chebconv_matrix, enable_bias, graph_conv_act_func):
        super(ChebConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.chebconv_matrix = chebconv_matrix
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.weight = nn.Parameter(torch.FloatTensor(self.Ks, self.c_in, self.c_out))
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(self.c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        # For Sigmoid, Tanh or Softsign
        if self.graph_conv_act_func == 'sigmoid' or self.graph_conv_act_func == 'tanh' or self.graph_conv_act_func == 'softsign':
            init.xavier_uniform_(self.weight)

        # For ReLU, Softplus, Leaky ReLU, PReLU, or ELU
        elif self.graph_conv_act_func == 'relu' or self.graph_conv_act_func == 'softplus' or self.graph_conv_act_func == 'leakyrelu' \
            or self.graph_conv_act_func == 'prelu' or self.graph_conv_act_func == 'elu':
            init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        # Using recurrence relation to reduce time complexity from O(n^2) to O(K|E|),
        # where K = Ks - 1
        x = x.reshape(n_vertex, -1)
        x_0 = x
        x_1 = torch.mm(self.chebconv_matrix, x)
        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks must be greater than 0, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.mm(2 * self.chebconv_matrix, x_list[k - 1]) - x_list[k - 2])
        x_tensor = torch.stack(x_list, dim=0)

        x_mul = torch.mm(x_tensor.reshape(-1, self.Ks * c_in), self.weight.reshape(self.Ks * c_in, -1)).reshape(-1, self.c_out)

        if self.bias is not None:
            x_chebconv = x_mul + self.bias
        else:
            x_chebconv = x_mul
        
        return x_chebconv

class GCNConv(nn.Module):
    def __init__(self, c_in, c_out, gcnconv_matrix, enable_bias, graph_conv_act_func):
        super(GCNConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gcnconv_matrix = gcnconv_matrix
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.weight = nn.Parameter(torch.FloatTensor(self.c_in, self.c_out))
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(self.c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        # For Sigmoid, Tanh or Softsign
        if self.graph_conv_act_func == 'sigmoid' or self.graph_conv_act_func == 'tanh' or self.graph_conv_act_func == 'softsign':
            init.xavier_uniform_(self.weight)

        # For ReLU, Softplus, Leaky ReLU, PReLU, or ELU
        elif self.graph_conv_act_func == 'relu' or self.graph_conv_act_func == 'softplus' or self.graph_conv_act_func == 'leakyrelu' \
            or self.graph_conv_act_func == 'prelu' or self.graph_conv_act_func == 'elu':
            init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        x_first_mul = torch.mm(x.reshape(-1, c_in), self.weight).reshape(n_vertex, -1)
        x_second_mul = torch.mm(self.gcnconv_matrix, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_gcnconv_out = x_second_mul + self.bias
        else:
            x_gcnconv_out = x_second_mul
        
        return x_gcnconv_out

class GraphConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, graph_conv_type, graph_conv_matrix, graph_conv_act_func):
        super(GraphConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(self.c_in, self.c_out)
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.graph_conv_act_func = graph_conv_act_func
        self.enable_bias = True
        if self.graph_conv_type == "chebconv":
            self.chebconv = ChebConv(self.c_out, self.c_out, self.Ks, self.graph_conv_matrix, self.enable_bias, self.graph_conv_act_func)
        elif self.graph_conv_type == "gcnconv":
            self.gcnconv = GCNConv(self.c_out, self.c_out, self.graph_conv_matrix, self.enable_bias, self.graph_conv_act_func)

    def forward(self, x):
        x_gc_in = self.align(x)
        batch_size, c_in, T, n_vertex = x_gc_in.shape
        if self.graph_conv_type == "chebconv":
            x_gc = self.chebconv(x_gc_in)
        elif self.graph_conv_type == "gcnconv":
            x_gc = self.gcnconv(x_gc_in)
        x_gc_with_rc = torch.add(x_gc.reshape(batch_size, self.c_out, T, n_vertex), x_gc_in)
        x_gc_out = x_gc_with_rc
        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv or GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, gated_act_func, graph_conv_type, graph_conv_matrix, drop_rate):
        super(STConvBlock, self).__init__()
        self.Kt = Kt
        self.Ks = Ks
        self.n_vertex = n_vertex
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.graph_conv_act_func = 'relu'
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Kt, self.last_block_channel, self.channels[0], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.graph_conv = GraphConvLayer(self.Ks, self.channels[0], self.channels[1], self.graph_conv_type, self.graph_conv_matrix, self.graph_conv_act_func)
        self.tmp_conv2 = TemporalConvLayer(self.Kt, self.channels[1], self.channels[2], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.tc2_ln = nn.LayerNorm([self.n_vertex, self.channels[2]])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)
        x_graph_conv = self.graph_conv(x_tmp_conv1)
        if self.graph_conv_act_func == 'sigmoid':
            x_act_func = self.sigmoid(x_graph_conv)
        elif self.graph_conv_act_func == 'tanh':
            x_act_func = self.tanh(x_graph_conv)
        elif self.graph_conv_act_func == 'softsign':
            x_act_func = self.softsign(x_graph_conv)
        elif self.graph_conv_act_func == 'relu':
            x_act_func = self.relu(x_graph_conv)
        elif self.graph_conv_act_func == 'softplus':
            x_act_func = self.softplus(x_graph_conv)
        elif self.graph_conv_act_func == 'leakyrelu':
            x_act_func = self.leakyrelu(x_graph_conv)
        elif self.graph_conv_act_func == 'prelu':
            x_act_func = self.prelu(x_graph_conv)
        elif self.graph_conv_act_func == 'elu':
            x_act_func = self.elu(x_graph_conv)
        x_tmp_conv2 = self.tmp_conv2(x_act_func)
        x_tc2_ln = self.tc2_ln(x_tmp_conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_do = self.do(x_tc2_ln)
        x_st_conv_out = x_do
        return x_st_conv_out

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, gated_act_func, drop_rate):
        super(OutputBlock, self).__init__()
        self.Ko = Ko
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.end_channel = end_channel
        self.n_vertex = n_vertex
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Ko, self.last_block_channel, self.channels[0], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.fc1 = nn.Linear(self.channels[0], self.channels[1])
        self.fc2 = nn.Linear(self.channels[1], self.end_channel)
        self.tc1_ln = nn.LayerNorm([self.n_vertex, self.channels[0]])
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
        x_tc1 = self.tmp_conv1(x)
        x_tc1_ln = self.tc1_ln(x_tc1.permute(0, 2, 3, 1))
        x_fc1 = self.fc1(x_tc1_ln)
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
