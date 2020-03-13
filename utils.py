import torch 
import pandas as pd 
import numpy as np 
import pickle

TRAINDATA = 'all_jets_train_4D_100_percent.pkl'
TESTDATA = 'all_jets_test_4D_100_percent.pkl'
N_TRAIN = 111778
N_TEST = 27945
def get_data_batches(n_train=90000, n_val=21778, device='cuda', frac=0.1):
    assert n_train + n_val == N_TRAIN
    with open(TRAINDATA, 'rb') as f:
        traindata = pickle.load(f).to_numpy()
    with open(TESTDATA, 'rb') as f:
        testdata = pickle.load(f).to_numpy()
    train_len = traindata.shape[0]
    assert train_len == N_TRAIN
    test_len = testdata.shape[0]
    assert test_len == N_TEST

    traindata = traindata[:int(N_TRAIN*frac)]
    testdata = testdata[:int(N_TEST*frac)]
    mean = traindata.mean(axis=0)
    std = traindata.std(axis=0)

    train_min = traindata.min(axis=0)
    train_max = traindata.max(axis=0)
    """
    traindata = (traindata - mean) / std 
    testdata = (testdata - mean) / std
    """
    traindata = (traindata - train_min) / (train_max - train_min)
    testdata = (testdata - train_min) / (train_max - train_min)
    np.random.shuffle(traindata)
    np.random.shuffle(testdata)

    print(traindata.mean(), traindata.std())
    print(testdata.mean(), testdata.std())

    train_batch = torch.FloatTensor(traindata[:int(n_train*frac)]).to(device)
    val_batch = torch.FloatTensor(traindata[int(n_train*frac):int(train_len*frac)]).to(device)
    test_batch = torch.FloatTensor(testdata[:int(test_len*frac)]).to(device)

    return train_batch, val_batch, test_batch

def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))
