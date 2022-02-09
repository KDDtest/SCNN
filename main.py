# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time

import numpy as np
import math
from models.Wavenet import Wavenet
from utils.data_utils import *
from utils.math_utils import *
from utils.tester import model_inference
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse


torch.backends.cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


batch_size = 8  # batch size
test_batch_size = 8

lr = 0.0001  # learning rate


import os
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

parser = argparse.ArgumentParser()
parser.add_argument('-version', type=int, default=0)
parser.add_argument('-mode', type=str, default='train')
parser.add_argument('-n_his', type=int, default=144)
parser.add_argument('-n_pred', type=int, default=3)
parser.add_argument('-n_layers', type=int, default=4)

parser.add_argument('-normA', type=int, default=1)
parser.add_argument('-normB', type=int, default=1)
parser.add_argument('-normC', type=int, default=1)
parser.add_argument('-normD', type=int, default=1)
parser.add_argument('-hidden_channels', type=int, default=16)
parser.add_argument('-dataset', type=str, default='bike')

parser.add_argument('cuda',type=int,default=3,help='cuda device id')
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")

n_his = args.n_his
n_pred = args.n_pred
n_layers = args.n_layers
hidden_channels = args.hidden_channels
normA_bool = args.normA
normB_bool = args.normB
normC_bool = args.normC
normD_bool = args.normD

dataset_name = args.dataset
version = args.version
target_fname = f"{dataset_name}_hc{hidden_channels}_l{n_layers}_h{n_his}_pred{n_pred}_normA{normA_bool}_normB{normB_bool}_normC{normC_bool}_normD{normD_bool}_v{version}"

best_model_path = os.path.join('MODEL', f"{target_fname}_best.h5")


def predict(pred_model, dataset, n, partition='test', epoch=None):

    print("predicting...")

    x_stats = dataset.get_stats()
    #pred_norms = []
    torch.cuda.empty_cache()

    pred_model.eval()
    preds = []
    reps = []
    ys = []
    xs = []
    criterion = nn.MSELoss()
    for x_batch, time_batch in gen_batch(dataset.get_data(partition), test_batch_size, dynamic_batch=True, shuffle=False):
        xh = x_batch[:, :-n_pred]
        y = x_batch[:, -n_pred:]
        timeh = time_batch[:, :-n_pred]
        xh = torch.tensor(xh, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        timeh = torch.tensor(timeh, dtype=torch.int64).to(device)

        pred = pred_model(xh, timeh)
        reps.append(rep.data.cpu().numpy())
        pred = pred[..., -1:]
        preds.append(pred.data.cpu().numpy())
        ys.append(y.data.cpu().numpy())
        xs.append(xh.data.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    xs = np.concatenate(xs, axis=0)
    ys = z_inverse(ys, x_stats['mean'], x_stats['std'])
    xs = z_inverse(xs, x_stats['mean'], x_stats['std'])
    preds = z_inverse(preds, x_stats['mean'], x_stats['std'])
    np.save(f"RESULT/{target_fname}_{partition}_y.npy", ys)
    np.save(f"RESULT/{target_fname}_{partition}_x.npy", xs)
    np.save(f"RESULT/{target_fname}_{partition}_pred.npy", preds)


def train(model, dataset, n):
    print('=' * 10)
    print("training model...")

    print("releasing gpu memory....")
    model.train()
    l2_criterion = nn.MSELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, [p for p in model.parameters()]), lr=lr)
    torch.cuda.empty_cache()
    min_rmse = 1000
    min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
    stop = 0
    nb_epoch = 1000

    for epoch in range(nb_epoch):  # loop over the dataset multiple times  # nb_epoch
        model.train()
        running_loss = []
        for j, (x_batch, time_batch) in enumerate(gen_batch(dataset.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
            xh = x_batch[:, :-n_pred]
            timeh = time_batch[:, :-n_pred]
            y = x_batch[:, -n_pred:]

            xh = torch.tensor(xh, dtype=torch.float32).to(device)
            timeh = torch.tensor(timeh, dtype=torch.int64).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            pred = model(xh, timeh)

            model.zero_grad()
            loss = l2_criterion(pred[..., -1:], y).mean()
            loss.backward()
            running_loss.append(loss.data.cpu().numpy())
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            min_va_val, min_val, min_ta_val = model_inference(device, model, dataset, test_batch_size, n_his, n_pred, min_va_val, min_val, n)
            print(f'Epoch {epoch}:')
            va, te, ta = min_va_val, min_val, min_ta_val
            for i in range(n_pred):
                print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%}, {ta[i*3]:7.3%};'
                      f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f}, {ta[i*3+1]:4.3f};'
                      f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}, {ta[i*3+2]:6.3f}.')


            total_rmse = np.sum([va[i*3+2] for i in range(n_pred)])
            if total_rmse < min_rmse:
                torch.save(model.state_dict(), best_model_path)
                min_rmse = total_rmse
                stop = 0
            else:
                stop += 1
            if stop == 10:
                break
    model.load_my_state_dict(torch.load(best_model_path))
    min_va_val, min_val, ta = model_inference(device, model, dataset, test_batch_size, n_his, n_pred, min_va_val, min_val, n)
    va, te = min_va_val, min_val
    print('Best Results:')
    for i in range(n_pred):
        print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%}, {ta[i*3]:7.3%};'
            f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f}, {ta[i*3+1]:4.3f};'
            f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}, {ta[i*3+2]:6.3f}.')
    mean_va_mape = np.mean([va[i*3] for i in range(n_pred)])
    mean_te_mape = np.mean([te[i*3] for i in range(n_pred)])
    mean_va_mae = np.mean([va[i*3+1] for i in range(n_pred)])
    mean_te_mae = np.mean([te[i*3+1] for i in range(n_pred)])
    mean_va_rmse = np.mean([va[i*3+2] for i in range(n_pred)])
    mean_te_rmse = np.mean([te[i*3+2] for i in range(n_pred)])

    print(f'MAPE {mean_va_mape:7.3%}, {mean_te_mape:7.3%};'
        f'MAE  {mean_va_mae:4.3f}, {mean_te_mae:4.3f};'
        f'RMSE {mean_va_rmse:6.3f}, {mean_te_rmse:6.3f}.')


def eval(model, dataset, n, versions):
    print('=' * 10)
    print("evaluating model...")
    vas = []
    tes = []
    for _v in versions:
        torch.cuda.empty_cache()
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
        target_fname = f"{dataset_name}_hc{hidden_channels}_l{n_layers}_h{n_his}_pred{n_pred}_normA{normA_bool}_normB{normB_bool}_normC{normC_bool}_normD{normD_bool}_v{_v}"
        target_model_path = os.path.join('MODEL', '{}_best.h5'.format(target_fname))
        if os.path.isfile(target_model_path):
            model.load_my_state_dict(torch.load(target_model_path))
        else:
            print("file not exist")
            break
        min_va_val, min_val, _ = model_inference(device, model, dataset, test_batch_size, n_his, n_pred, min_va_val, min_val, n)
        print(f'Version:{_v}')
        va, te = min_va_val, min_val
        for i in range(n_pred):
            print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')
        vas.append(va)
        tes.append(te)
    va = np.array(vas).mean(axis=0)
    te = np.array(tes).mean(axis=0)
    print(f'Overall:')
    for i in range(n_pred):
        print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
            f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
            f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')

    mean_va_mape = np.mean([va[i*3] for i in range(n_pred)])
    mean_te_mape = np.mean([te[i*3] for i in range(n_pred)])
    mean_va_mae = np.mean([va[i*3+1] for i in range(n_pred)])
    mean_te_mae = np.mean([te[i*3+1] for i in range(n_pred)])
    mean_va_rmse = np.mean([va[i*3+2] for i in range(n_pred)])
    mean_te_rmse = np.mean([te[i*3+2] for i in range(n_pred)])

    print(f'MAPE {mean_va_mape:7.3%}, {mean_te_mape:7.3%};'
        f'MAE  {mean_va_mae:4.3f}, {mean_te_mae:4.3f};'
        f'RMSE {mean_va_rmse:6.3f}, {mean_te_rmse:6.3f}.')


def main():
    # load data
    print("loading data...")
    if dataset_name == 'bike':
        n_train, n_val, n_test = 163, 10, 10
        n = 128
        n_slots = 24
        dataset = data_gen('data/NYC14_bike.csv', (n_train, n_val, n_test), n, n_his + n_pred, n_slots)
    if dataset_name == 'elec':
        n_train, n_val, n_test = 78, 7, 7
        n = 336
        n_slots = 24
        dataset = data_gen('data/electricity.csv', (n_train, n_val, n_test), n, n_his + n_pred, n_slots)
    if dataset_name == 'pems':
        n_train, n_val, n_test = 34, 5, 5
        n = 228
        n_slots = 48
        dataset = data_gen('data/PeMS.csv', (n_train, n_val, n_test), n, n_his + n_pred, n_slots)
    print('=' * 10)
    print("compiling model...")

    model = Wavenet(device, n, n_pred, n_his, n_slots, normA_bool=normA_bool, normB_bool=normB_bool, normC_bool=normC_bool, normD_bool=normD_bool, in_dim=1, out_dim=1, channels=hidden_channels, layers=n_layers).to(device)
    print('=' * 10)

    for name, p in model.named_parameters():
        name = name.split('.')
        if name[-1] in ['gamma']:
            continue
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    if args.mode == 'train':
        train(model, dataset, n)
    if args.mode == 'eval':
        eval(model, dataset, n, range(5))      
    if args.mode == 'pred':
        model.load_my_state_dict(torch.load(best_model_path))
        predict(model, dataset, n)

if __name__ == '__main__':
    main()
