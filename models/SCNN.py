import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from pts.modules import StudentTOutput, NormalOutput
import random

epsilon = 1

        
def SeasonalNorm(x, period_length):
    b, c, n, t = x.shape
    x_period = torch.split(x, split_size_or_sections=period_length, dim=-1)
    x_period = torch.stack(x_period, -2)

    mean = x_period.mean(3)
    var = (x_period ** 2).mean(3) - mean ** 2 + 0.00001

    mean = mean.repeat(1, 1, 1, t // period_length)
    var = var.repeat(1, 1, 1, t // period_length)

    mean = F.pad(mean.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1) 
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5
    

class AdaSpatialNorm(nn.Module):
    def __init__(self, device, embedding_dim, num_nodes):
        super(AdaSpatialNorm, self).__init__()
        self.device = device
        self.node_embedding = nn.Parameter(torch.zeros(num_nodes, embedding_dim))
        
    def forward(self, x):
        b, c, n, t = x.shape
        
        adj_mat = torch.matmul(self.node_embedding, self.node_embedding.T)
        adj_mat = adj_mat - 10 * torch.eye(n).to(self.device)
        adj_mat = torch.softmax(adj_mat, dim=-1)

        adj_mat = adj_mat.unsqueeze(0)
        x_f = x.permute(0, 3, 2, 1).reshape(b * t, -1, c)
        
        mean_f = torch.matmul(adj_mat, x_f)
        var_f = torch.matmul(adj_mat, x_f ** 2) - mean_f ** 2 + 0.00001
          
        mean = mean_f.view(b, t, n, c).permute(0, 3, 2, 1)
        var = var_f.view(b, t, n, c).permute(0, 3, 2, 1)        
        
        out = (x - mean) / (var + epsilon) ** 0.5

        return out, mean, var ** 0.5


def TermNorm(x, term_length):
    b, c, n, t = x.shape
    x_patchify = [x[..., term_length-1-i:-i+t] for i in range(0, term_length)]
    x_patchify = torch.stack(x_patchify, dim=-1)

    mean = x_patchify.mean(4)
    var = (x_patchify ** 2).mean(4) - mean ** 2 + 0.00001
    mean = F.pad(mean.reshape(b * c, n, -1), mode='replicate', pad=(term_length-1, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='replicate', pad=(term_length-1, 0)).reshape(b, c, n, -1)
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5


class ResidualExtrapolate(nn.Module):
    def __init__(self, channels, num_input, num_output):
        super(ResidualExtrapolate, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.proj_map = nn.Conv2d(in_channels=channels, out_channels=channels * num_output, kernel_size=(1, num_input))
        
    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.proj_map(x[..., -self.num_input:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)
        
        return x_proj


def SeasonalExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='circular', pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj


def ConstantExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='replicate', pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj


class SCLayer(nn.Module):
    def __init__(self, device, num_local_input, num_pred, num_nodes, period_length, short_term, long_term, channels, skip_channels, embedding_dim):
        super(SCLayer, self).__init__()
        self.device = device
        self.num_local_input = num_local_input
        self.num_pred = num_pred
        self.num_nodes = num_nodes
        self.period_length = period_length
        self.short_term = short_term
        self.long_term = long_term
        self.embedding_dim = embedding_dim
        
        self.spatial_norm = AdaSpatialNorm(device, embedding_dim, num_nodes)
        self.residual_extrapolate_1 = ResidualExtrapolate(channels, 5, num_pred)
        self.residual_extrapolate_2 = ResidualExtrapolate(channels, 5, num_pred)
        self.residual_extrapolate_3 = ResidualExtrapolate(channels, 5, num_pred)
        self.residual_extrapolate_4 = ResidualExtrapolate(channels, 5, num_pred)
        
        self.conv_1 = nn.Conv2d(in_channels=13 * channels, out_channels=channels, kernel_size=(1, num_local_input), dilation=1)
        self.conv_2 = nn.Conv2d(in_channels=13 * channels, out_channels=channels, kernel_size=(1, num_local_input), dilation=1)
        
        self.skip_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.scale_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.residual_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        b, c, n, t = x.shape
        residual = x
        xs = []
        x_aux = []
        z0 = x[..., -1]
        
        x_proj = ConstantExtrapolate(x, self.num_pred)
        xs.append(x_proj)
        x_aux.append(torch.zeros(x_proj.shape).to(self.device))
        
        x, long_term_mean, long_term_std = TermNorm(x, self.long_term)
        x_proj = self.residual_extrapolate_1(x)
        long_term_mean_proj, long_term_std_proj = ConstantExtrapolate(long_term_mean, self.num_pred), ConstantExtrapolate(long_term_std, self.num_pred)
        xs.extend([x_proj, long_term_mean_proj, long_term_std_proj])
        x_aux.extend([torch.zeros(x_proj.shape).to(self.device), long_term_mean_proj, long_term_std_proj])
        
        x, season_mean, season_std = SeasonalNorm(x, self.period_length)
        x_proj = self.residual_extrapolate_2(x)
        season_mean_proj, season_std_proj = SeasonalExtrapolate(season_mean, self.num_pred), SeasonalExtrapolate(season_std, self.num_pred)
        xs.extend([x_proj, season_mean_proj, season_std_proj])
        x_aux.extend([torch.zeros(x_proj.shape).to(self.device), season_mean_proj, season_std_proj])
        
        x, short_term_mean, short_term_std = TermNorm(x, self.short_term)
        x_proj = self.residual_extrapolate_3(x)
        short_term_mean_proj, short_term_std_proj = ConstantExtrapolate(short_term_mean, self.num_pred), ConstantExtrapolate(short_term_std, self.num_pred)
        xs.extend([x_proj, short_term_mean_proj, short_term_std_proj])
        x_aux.extend([torch.zeros(x_proj.shape).to(self.device), short_term_mean_proj, short_term_std_proj])
        
        x, spatial_mean, spatial_std = self.spatial_norm(x)
        x_proj = self.residual_extrapolate_4(x)
        spatial_mean_proj, spatial_std_proj = ConstantExtrapolate(spatial_mean, self.num_pred), ConstantExtrapolate(spatial_std, self.num_pred)
        xs.extend([x_proj, spatial_mean_proj, spatial_std_proj])
        x_aux.extend([torch.zeros(x_proj.shape).to(self.device), spatial_mean_proj, spatial_std_proj])        

        x = torch.cat(xs, dim=1)
        x_aux = torch.cat(x_aux, dim=1)
        
        x = F.pad(x, mode='constant', pad=(1, 0))
        x_aux = F.pad(x_aux, mode='constant', pad=(1, 0))

        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_aux_1 = self.conv_1(x_aux)
        x_aux_2 = self.conv_2(x_aux)
        
        x_z = (x_1 * x_2)[..., :-self.num_pred]
        pred_z = (x_1 * x_2)[..., -self.num_pred:]
        pred_aux_z = (x_aux_1 * x_aux_2)[..., -self.num_pred:]
        s = self.skip_conv(pred_z)
        s_aux = self.skip_conv(pred_aux_z)
        x_z = self.residual_conv(x_z)

        return x_z, s_aux, s


class SCNN(nn.Module):
    def __init__(self, device, num_local_input, num_pred, num_nodes, period_length, short_term, long_term, channels, in_dim=1, out_dim=1, layers=2):
        super(SCNN, self).__init__()
        self.layers = layers
        self.device = device
        self.sc_layers = nn.ModuleList()
        self.num_pred = num_pred
        self.num_nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=channels,
                                    kernel_size=1)
        self.skip_channels = channels
        for i in range(layers):
            self.sc_layers.append(SCLayer(device, num_local_input, num_pred, num_nodes, period_length, short_term, long_term, channels, channels, channels))

        self.end_conv = nn.Conv1d(in_channels=num_pred * self.skip_channels,
                                  out_channels=num_pred * self.skip_channels,
                                  groups=num_pred,
                                  kernel_size=1,
                                  bias=True)
        self.distr_output = StudentTOutput()
        self.proj_distr_args = self.distr_output.get_args_proj(self.skip_channels)

    def forward(self, input):
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        reps = []
        rep_ys = []
        xs = []
        skip = 0
        
        x = self.start_conv(input)
        b, c, n, t = x.shape
        out = 0
        out_aux = 0
        s = 0

        for i in range(self.layers):
            residual = x
            xs.append(x[..., -1:])
            x, s_aux, s = self.sc_layers[i](x)
            out = s + out
            out_aux = s_aux + out_aux

        xs = torch.cat(xs, dim=-1)
        out = out.permute(0, 3, 1, 2).reshape(b, -1, n)
        out_aux = out_aux.permute(0, 3, 1, 2).reshape(b, -1, n)
        out = self.end_conv(out).reshape(b, self.num_pred, self.skip_channels, n).permute(0, 3, 1, 2).reshape(b * n, self.num_pred, self.skip_channels)
        out_aux = self.end_conv(out_aux).reshape(b, self.num_pred, self.skip_channels, n).permute(0, 3, 1, 2).reshape(b * n, self.num_pred, self.skip_channels)
        
        pred_distr_args = self.proj_distr_args(out)
        pred_distr_args = (pred_distr_args[0], pred_distr_args[1], F.threshold(pred_distr_args[2], 0.2, 0.2))
        pred_distr = self.distr_output.distribution(pred_distr_args)
        pred_distr_args_aux = self.proj_distr_args(out_aux)
        pred_distr_args_aux = (pred_distr_args_aux[0], pred_distr_args_aux[1], F.threshold(pred_distr_args_aux[2], 0.2, 0.2))
        pred_distr_aux = self.distr_output.distribution(pred_distr_args_aux)
        pred_mean = pred_distr_args[1].reshape(b, n, self.num_pred, 1).permute(0, 2, 1, 3)
        if self.training:
            return pred_distr, pred_distr_aux
        else:
            return pred_mean

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print(name, param.shape)
                pass


