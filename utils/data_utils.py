import os
from .math_utils import z_score
import h5py
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats, period=None):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    n_slot = day_slot

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            end = (i + offset) * day_slot + j + 1
            sta = end - n_frame
            if sta >= 0:
                tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])

    return tmp_seq


def mask_gen(file_path, data, threshold=1.0):
    if os.path.exists(file_path):
        mask = np.load(file_path)
    else:
        values = np.random.uniform(0, 1, data.shape)
        mask = (values > threshold).astype(np.int64)
        np.save(file_path, mask)
    return mask


def data_gen(data_file_path, data_config, n_route, n_frame=21, day_slot=288, period=24):

    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(data_file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    time_seq = np.arange(data_seq.shape[0]) % day_slot
    time_seq = time_seq.reshape((-1, 1))

    data_seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    time_seq_train = seq_gen(n_train, time_seq, 0, n_frame, 1, day_slot)
    data_seq_train = data_seq_train[n_frame:]
    time_seq_train = time_seq_train[n_frame:]
    data_seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    time_seq_val = seq_gen(n_val, time_seq, n_train, n_frame, 1, day_slot)
    data_seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)
    time_seq_test = seq_gen(n_test, time_seq, n_train + n_val, n_frame, 1, day_slot)


    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(data_seq_train, axis=(0, 1, 3)), 'std': np.std(data_seq_train, axis=(0, 1, 3))+0.0001}
    #x_stats = {'mean': np.mean(data_seq_train), 'std': np.std(data_seq_train)}
    x_train = z_score(data_seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(data_seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(data_seq_test, x_stats['mean'], x_stats['std'])
    x_data = {'train': [x_train, time_seq_train], 'val': [x_val, time_seq_val], 'test': [x_test, time_seq_test]}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    inputs, time_inputs = inputs
    num_node = inputs.shape[2]
    num_his = inputs.shape[1] - 3

    len_inputs = len(inputs)
    #if training and shuffle:
    #    offset = np.random.randint(0, 128, 1)[0]
    #    idx = np.arange(offset, len_inputs, 128)
    #    #idx = np.arange(len_inputs)
    #    np.random.shuffle(idx)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)
    else:
        idx = np.arange(len_inputs)

    idx_len = idx.shape[0]
    for start_idx in range(0, idx_len, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > idx_len:
            if dynamic_batch:
                end_idx = idx_len
            else:
                continue
        slide = idx[start_idx:end_idx]

        yield inputs[slide], time_inputs[slide]
