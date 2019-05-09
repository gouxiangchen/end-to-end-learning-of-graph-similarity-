import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import math
import re



class myLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
        # print('weight: ', self.weight)

    def reset_parameters(self):
        self.weight.data.normal_()

    def forward(self, A, D, H):
        # print('D:', D)
        D_ = torch.rsqrt(D)
        D_ = torch.diag(torch.diag(D_))
        # print('D_: ', D_)
        T = torch.mm(D_, A)
        # print('T1: ', T)
        T = torch.mm(T, D_)
        # print('T2: ', T)
        T = torch.mm(T, H)
        # print('T3: ', T)
        T = torch.mm(T, self.weight)


        # print(norm)
        # print('T4: ', T)
        # f = open('our.txt', 'a+')
        # print(self.weight, file=f)
        # f.close()
        return T


class combineLayer(nn.Module):
    def __init__(self, N, K):
        super(combineLayer, self).__init__()
        self.N = N
        self.K = K
        self.weight1 = nn.Parameter(torch.Tensor(N, N, K))
        self.weight2 = nn.Parameter(torch.Tensor(K, 2 * N))
        self.bias = nn.Parameter(torch.Tensor(K, 1))
        self.reset_parameters()
        # print('weight: ', self.weight)

    def reset_parameters(self):
        self.weight1.data.normal_()
        self.weight2.data.normal_()
        self.bias.data.normal_()

    def forward(self, X, Y):
        # print(self.weight1.size())
        first_weights = self.weight1.split(1, 2)
        # print(first_weights[0].size())
        r = torch.Tensor()
        for weight in first_weights:
            weight = weight.squeeze()
            M = torch.mm(X, weight)
            M = torch.mm(M, torch.t(Y))
            r = torch.cat((r, M), 0)
        # print(X.size(), Y.size())
        com_x_y = torch.cat((X, Y), 1)
        com_x_y = com_x_y.squeeze().unsqueeze(1)
        # print(com_x_y.size())
        p = torch.mm(self.weight2, com_x_y)
        r = r + p
        r = r + self.bias
        return r


class graphConvNet(nn.Module):
    def __init__(self):
        super(graphConvNet, self).__init__()
        self.gc1 = myLayer(128, 64)
        self.gc2 = myLayer(64, 32)
        self.gc3 = myLayer(32, 16)
        self.ReLU = nn.ReLU()
        self.combine = combineLayer(16, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A1, D1, H1, A2, D2, H2):
        # print('A: ', A, '\n|| D: ', D, '\n|| H: ', H)
        X = self.gc1(A1, D1, H1)
        X = self.ReLU(X)
        # print('first gc :', X, X.size())
        X = self.gc2(A1, D1, X)
        X = self.ReLU(X)
        # print('second gc :', X, X.size())
        X = self.gc3(A1, D1, X)

        # norm = []
        # for i in range(X.size(1)):
        #     s = torch.index_select(X, 1, torch.LongTensor([i]))
        #     sum_col = float(s.sum().data.numpy())
        #     sum_col = X.size(1) / sum_col
        #     norm.append(sum_col)
        # norm_tensor = torch.diag(torch.Tensor(norm))
        # X = torch.mm(X, norm_tensor)

        X = self.ReLU(X)
        # print('third gc :', X, X.size())

        Y = self.gc1(A2, D2, H2)
        Y = self.ReLU(Y)
        Y = self.gc2(A2, D2, Y)
        Y = self.ReLU(Y)
        Y = self.gc3(A2, D2, Y)

        # norm = []
        # for i in range(Y.size(1)):
        #     s = torch.index_select(Y, 1, torch.LongTensor([i]))
        #     sum_col = float(s.sum().data.numpy())
        #     sum_col = Y.size(1) / sum_col
        #     norm.append(sum_col)
        # norm_tensor = torch.diag(torch.Tensor(norm))
        # Y = torch.mm(Y, norm_tensor)

        Y = self.ReLU(Y)

        # print('X begin pooled :', X, 'Y begin pooled :', Y)

        Nx = X.size(0)
        Ny = Y.size(0)
        X_po = torch.Tensor([1 for i in range(Nx)])
        Y_po = torch.Tensor([1 for i in range(Ny)])
        X_po = X_po.unsqueeze(0)
        Y_po = Y_po.unsqueeze(0)
        X = torch.mm(X_po, X)
        Y = torch.mm(Y_po, Y)
        X = torch.div(X, Nx)
        Y = torch.div(Y, Ny)

        # print('X pooled :', X, 'Y pooled :', Y)
        # dist = np.linalg.norm(X.data.numpy() - Y.data.numpy())
        # return dist

        Z = self.combine(X, Y)
        # print(Z.size())
        Z = Z.squeeze()
        Z = self.ReLU(Z)
        # print('first sigmoid : ', Z)
        Z = self.fc1(Z)
        Z = self.ReLU(Z)
        # print('second sigmoid : ', Z)
        Z = self.fc2(Z)
        Z = self.ReLU(Z)
        # print('third sigmoid : ', Z)
        Z = self.fc3(Z)

        Z = self.ReLU(Z)
        # print(Z)
        Z = Z.squeeze()
        # print(Z.size())
        return Z



mynet = graphConvNet()
mynet.load_state_dict(torch.load('huge_dataset_model_total.para'))
filename_x = './graph_samples/00sheep0.txt'
filename_y = './graph_samples/1crime.txt'


G1 = nx.read_edgelist(filename_x)
G2 = nx.read_edgelist(filename_y)
A1 = torch.Tensor(nx.to_numpy_matrix(G1))
A2 = torch.Tensor(nx.to_numpy_matrix(G2))
H1 = torch.eye(A1.size(0), 128)
H2 = torch.eye(A2.size(0), 128)
list1 = []
list2 = []
for node in G1.nodes:
    list1.append(G1.degree(node))
for node in G2.nodes:
    list2.append(G2.degree(node))
D1 = torch.diag(torch.Tensor(list1))
D2 = torch.diag(torch.Tensor(list2))
out = mynet(A1, D1, H1, A2, D2, H2)

print('the GFD distance predicted between ' + filename_x + ' and ' + filename_y + ' is : ', float(out.data))





