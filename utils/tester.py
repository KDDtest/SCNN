
from .data_utils import *
from .math_utils import evaluation
from os.path import join as pjoin
import numpy as np
import torch.nn.functional as F
import time
import torch
from sklearn.linear_model import LinearRegression


def multi_pred(device, model, seq, batch_size, num_his, n_pred, dynamic_batch=True):
    pred_list = []
    model.eval()

    for i, time in gen_batch(seq, min(batch_size, len(seq[0])), shuffle=False, dynamic_batch=True):
        test_seq = np.copy(i[:, 0:num_his, :, :])
        time_seq = np.copy(time[:, 0:num_his, :, :])
        xh = torch.tensor(test_seq, dtype=torch.float32).to(device)
        timeh = torch.tensor(time_seq, dtype=torch.int64).to(device)
        y = i[:, num_his:num_his + n_pred]
        y = torch.tensor(y, dtype=torch.float32).to(device)
        pred = model(xh, timeh)
        pred = pred[..., -1:]
        pred = pred.data.cpu().numpy()
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def model_inference(device, model, inputs, batch_size, num_his, n_pred, min_va_val, min_val, n):
    x_train, x_val, x_test, x_stats = inputs.get_data('train'), inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    if num_his + n_pred > x_val[0].shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')
    y_val, len_val = multi_pred(device, model, x_val, batch_size, num_his, n_pred)
    y_train, len_train = multi_pred(device, model, x_train, batch_size, num_his, n_pred)
    y_pred, len_pred = multi_pred(device, model, x_test, batch_size, num_his, n_pred)
    evl_val = evaluation(x_val[0][0:len_val, num_his:n_pred + num_his][:, :], y_val[:, :], x_stats)
    evl_train = evaluation(x_train[0][0:len_train, num_his:n_pred + num_his][:, :], y_train[:, :], x_stats)
    # update the metric on test set, if model's performance got improved on the validation.
    evl_pred = evaluation(x_test[0][0:len_pred, num_his:n_pred + num_his][:, :], y_pred[:, :], x_stats)

    min_val = evl_pred
    return evl_val, evl_pred, evl_train

