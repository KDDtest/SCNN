import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import sys


class NormA(nn.Module):
    def __init__(self, device, num_his):
        super(NormA, self).__init__()
        self.device = device
        self.weights = nn.Parameter(torch.zeros(num_his))

    def forward(self, x):
        b, c, n, t = x.shape
        weights = self.weights
        weights = torch.softmax(weights, dim=0).reshape(1, 1, -1)
        x_f = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        mean_f = torch.matmul(weights, x_f)
        var_f = torch.matmul(weights, x_f ** 2) - mean_f ** 2
        mean = mean_f.view(b, n, 1, c).permute(0, 3, 1, 2)
        var = var_f.view(b, n, 1, c).permute(0, 3, 1, 2)
        out = (x - mean) / (var + 0.00001) ** 0.5
        return out


class NormB(nn.Module):
    def __init__(self,device, num_nodes):
        super(NormB, self).__init__()
        self.device = device
        self.weights = nn.Parameter(torch.zeros(num_nodes))
        
    def forward(self, x):
        b, c, n, t = x.shape
        weights = self.weights
        weights = torch.softmax(weights, dim=0).reshape(1, 1, -1)
        x_f = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        mean = torch.matmul(weights, x_f)
        var = torch.matmul(weights, x_f ** 2) - mean ** 2 + 0.00001
        out_f = (x_f - mean) / var ** 0.5
        out = out_f.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return out
    

class NormC(nn.Module):
    def __init__(self, device, num_his):
        super(NormC, self).__init__()
        self.device = device
        self.weights = nn.Parameter(torch.zeros(num_his))

    def forward(self, x):
        b, c, n, t = x.shape
        weights = self.weights
        weights = torch.softmax(weights.repeat(n), dim=0).reshape(1, 1, -1)
        x_f = x.permute(0, 2, 3, 1).reshape(b, n * t, c)
        mean_f = torch.matmul(weights, x_f)
        var_f = torch.matmul(weights, x_f ** 2) - mean_f ** 2 + 0.00001
        mean = mean_f.permute(0, 2, 1).reshape(b, c, 1, 1)
        var = var_f.permute(0, 2, 1).reshape(b, c, 1, 1)
        out = (x - mean) / var ** 0.5
        return out


class NormD(nn.Module):
    def __init__(self, device, channels, num_slots=24, momentum=0.1):
        super(NormD, self).__init__()
        self.num_slots = num_slots
        self.register_buffer('running_mean', torch.zeros(1, num_slots, channels))
        self.register_buffer('running_var', torch.ones(1, num_slots, channels))
        self.momentum = momentum

    def forward(self, x, time):
        b, c, n, t = x.shape
        time = time[..., -t:]
        time_onehot = F.one_hot(time, num_classes=self.num_slots)
        time_onehot_f = time_onehot.reshape(b * t, -1)
        time_weight_f = F.normalize(torch.as_tensor(time_onehot_f, dtype=torch.float32), p=1, dim=0)
        x_f = x.permute(2, 0, 3, 1).reshape(n, b * t, c)
        mean_period = torch.matmul(time_weight_f.unsqueeze(0).permute(0, 2, 1), x_f).mean(0, keepdims=True)
        var_period = torch.matmul(time_weight_f.unsqueeze(0).permute(0, 2, 1), x_f ** 2).mean(0, keepdims=True) - mean_period ** 2
        if self.training:
            n = time_onehot_f.sum(0).reshape(1, -1, 1) * n
            with torch.no_grad():
                self.running_mean = self.momentum * mean_period + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var_period * n / (n - 1) + (1 - self.momentum) * self.running_var
        else:
            mean_period = self.running_mean
            var_period = self.running_var

        mean_period_f = mean_period.permute(1, 0, 2).reshape(self.num_slots, -1)
        var_period_f = var_period.permute(1, 0, 2).reshape(self.num_slots, -1)
        sample_mean = F.embedding(time, mean_period_f).reshape(b, t, -1, c).permute(0, 3, 2, 1)
        sample_var = F.embedding(time, var_period_f).reshape(b, t, -1, c).permute(0, 3, 2, 1)
        out = (x - sample_mean) / (sample_var + 0.00001) ** 0.5
        return out


class Affine(nn.Module):
    def __init__(self, channels):
        super(Affine, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        b, c, n, t = x.shape
        out = x * self.gamma + self.beta
        return out


class GLU(nn.Module):
    def __init__(self, device, num_nodes, num_pred, num_his, num_slots, channels, skip_channels, normA_bool, normB_bool, normC_bool, normD_bool):
        super(GLU, self).__init__()
        self.normA_bool = normA_bool
        self.normB_bool = normB_bool
        self.normC_bool = normC_bool
        self.normD_bool = normD_bool
        
        if self.normA_bool:
            self.normA = NormA(device, num_his)
            self.affineA = Affine(channels)
            
        if self.normB_bool:
            self.normB = NormB(device, num_nodes)
            self.affineB = Affine(channels)

        if self.normC_bool:
            self.normC = NormC(device, num_his)
            self.affineC = Affine(channels)
            
        if self.normD_bool:
            self.normD = NormD(device, channels)
            self.affineD = Affine(channels)

        num = 1 + self.normA_bool + self.normB_bool + self.normC_bool + self.normD_bool
        self.filter_conv = nn.Conv2d(in_channels=num*channels, out_channels=channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(in_channels=num*channels, out_channels=channels, kernel_size=1)

        self.residual_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(in_channels=channels, out_channels=skip_channels, kernel_size=1)

    def forward(self, x, time):
        residual = x
        x_list = [x]
        if self.normA_bool:
            x_a = self.normA(x)
            x_a = self.affineA(x_a)
            x_list.append(x_a)
        if self.normB_bool:
            x_b = self.normB(x)
            x_b = self.affineB(x_b)
            x_list.append(x_b)
        if self.normC_bool:
            x_c = self.normC(x)
            x_c = self.affineC(x_c)
            x_list.append(x_c)
        if self.normD_bool:
            x_d = self.normD(x, time)
            x_d = self.affineD(x_d)
            x_list.append(x_d)
        x_aug = torch.cat(x_list, dim=1)
        filter = self.filter_conv(x_aug)
        filter = torch.tanh(filter)

        gate = self.gate_conv(x_aug)
        gate = torch.sigmoid(gate)
        x = filter * gate
        s = x
        s = self.skip_conv(s)

        x = self.residual_conv(x)
        return x, s


class Wavenet(nn.Module):
    def __init__(self, device, num_nodes, num_pred, num_his, num_slots, normA_bool=False, normB_bool=False, normC_bool=False, normD_bool=False, in_dim=2,out_dim=2, channels=32, layers=2):
        super(Wavenet, self).__init__()
        self.layers = layers

        self.glu_blocks = nn.ModuleList()
        self.num_pred = num_pred
        self.num_nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=channels,
                                    kernel_size=1)
        skip_channels = channels

        for i in range(layers):
            self.glu_blocks.append(GLU(device, num_nodes, num_pred, num_his, num_slots, channels, skip_channels, normA_bool=normA_bool, normB_bool=normB_bool, normC_bool=normC_bool, normD_bool=normD_bool))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=skip_channels,
                                    kernel_size=1,
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=num_pred * out_dim,
                                    kernel_size=1,
                                    bias=True)

    def forward(self, input, time):
        input = input.permute(0, 3, 2, 1)
        time = time.permute(0, 3, 2, 1)
        in_len = input.size(3)

        skip = 0
        x = self.start_conv(input)

        for i in range(self.layers):
            residual = x
            x, s = self.glu_blocks[i](x, time)
            x = x + residual[..., -x.shape[-1]:]
            try:
                skip = s + skip[..., -s.shape[-1]:]
            except:
                skip = s

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)


