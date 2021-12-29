import numpy as np
import subprocess
import pickle
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_train1 = unpickle('cifar-10-batches-py/data_batch_1')
cifar10_train2 = unpickle('cifar-10-batches-py/data_batch_2')
cifar10_train3 = unpickle('cifar-10-batches-py/data_batch_3')
cifar10_train4 = unpickle('cifar-10-batches-py/data_batch_4')
cifar10_train5 = unpickle('cifar-10-batches-py/data_batch_5')

cifar10_test = unpickle('cifar-10-batches-py/test_batch')

x_tr1 = torch.from_numpy(cifar10_train1[b'data'])
x_tr2 = torch.from_numpy(cifar10_train2[b'data'])
x_tr3 = torch.from_numpy(cifar10_train3[b'data'])
x_tr4 = torch.from_numpy(cifar10_train4[b'data'])
x_tr5 = torch.from_numpy(cifar10_train5[b'data'])

y_tr1 = torch.LongTensor(cifar10_train1[b'labels'])
y_tr2 = torch.LongTensor(cifar10_train2[b'labels'])
y_tr3 = torch.LongTensor(cifar10_train3[b'labels'])
y_tr4 = torch.LongTensor(cifar10_train4[b'labels'])
y_tr5 = torch.LongTensor(cifar10_train5[b'labels'])

x_tr = torch.cat((x_tr1, x_tr2, x_tr3, x_tr4, x_tr5), dim=0)
y_tr = torch.cat((y_tr1, y_tr2, y_tr3, y_tr4, y_tr5), dim=0)

x_te = torch.from_numpy(cifar10_test[b'data'])
y_te = torch.LongTensor(cifar10_test[b'labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar10.pt')