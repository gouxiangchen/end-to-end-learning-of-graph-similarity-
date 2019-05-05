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


def get_dist(l1, l2):
    r = 0.
    for i in range(len(l1)):
        k = (l1[i] - l2[i]) * (l1[i] - l2[i])
        r += k
    r = math.sqrt(r)
    return r


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


def get_file_name(dir):
    list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            list.append(os.path.join(root, file))
    return list


mynet = graphConvNet()
mynet.load_state_dict(torch.load('huge_dataset_model_total.para'))
all_wrong = 0.
all_count = 0
filenames_x = get_file_name('../trainset-draw/')
filenames_y = get_file_name('../trainset-draw/')
# result_file = open('result_wrong_mixooo_128.txt', 'w')

draw_file = open('intrain.txt', 'w')


for ao in range(len(filenames_x)):
    # print(type(i))
    for pe in range(ao+1, len(filenames_x)):
        # print(ao, pe)
        filename_x = filenames_x[int(ao)]
        filename_y = filenames_x[int(pe)]
        f1 = filename_x.split('/')[-1]
        f2 = filename_y.split('/')[-1]
        fk1 = re.sub('[0-9]\d*', '', f1)
        fk2 = re.sub('[0-9]\d*', '', f2)

        # print(f1, f2)

        if False:
            pass
        else:
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
            label1 = []
            label2 = []
            la1 = f1.split('.')[0]
            la2 = f2.split('.')[0]
            label1_name = '../all_label/' + la1 + '_label.txt'
            label2_name = '../all_label/' + la2 + '_label.txt'
            # label2_name = 'E:\\chen\\图卷积——小图标签\\train_label.txt'

            file1 = open(label1_name, 'r')
            file2 = open(label2_name, 'r')

            for line in file1.readlines():
                i = line.split()[1]
                label1.append(int(i) + 1)
            for line in file2.readlines():
                i = line.split()[1]
                label2.append(int(i) + 1)

            label1 = np.array(label1)
            label1 = label1 / sum(label1)
            label1 = list(label1)

            label2 = np.array(label2)
            label2 = label2 / sum(label2)
            label2 = list(label2)

            label_1_log = []
            label_2_log = []
            for i in label1:
                i = math.log(i)
                label_1_log.append(i)
            for i in label2:
                i = math.log(i)
                label_2_log.append(i)

            l = get_dist(label_1_log, label_2_log)

            # print(la1, la2, float(out.data), l)
            # print(f1, f2, 'model out:', out, 'ground truth:', l, 'wrong rate:', abs(float(out.data) - l)/l, file=result_file)
            # if l == 0:
            #     print(float(out.data), l, filename_x, filename_y, file=draw_file)
            # else :
            #     print(float(out.data), l, file=draw_file)


            print(float(out.data), l, file=draw_file)


            # all_count += 1
            # all_wrong += abs(float(out.data) - l) / l
            file1.close()
            file2.close()


draw_file.close()
#
#
# filename_x = 'E:\\chen\\小图噪声\\mac.txt'
# filename_x = 'E:\\chen\\图卷积——小图\\sheep.txt'
# filename_y = 'E:\\chen\\图卷积——大图\\hypertext.txt'
# G1 = nx.read_edgelist(filename_x)
# G2 = nx.read_edgelist(filename_y)
# A1 = torch.Tensor(nx.to_numpy_matrix(G1))
# A2 = torch.Tensor(nx.to_numpy_matrix(G2))
# H1 = torch.eye(A1.size(0), 128)
# H2 = torch.eye(A2.size(0), 128)
# list1 = []
# list2 = []
# for node in G1.nodes:
#     list1.append(G1.degree(node))
# for node in G2.nodes:
#     list2.append(G2.degree(node))
# D1 = torch.diag(torch.Tensor(list1))
# D2 = torch.diag(torch.Tensor(list2))
# out = mynet(A1, D1, H1, A2, D2, H2)
# print(out)
#
# label1 = []
# label2 = []
# label1_name = 'E:\\chen\\git\\graphlet_counting-master\\hypertext_label.txt'
# label2_name = 'E:\\chen\\图卷积——小图标签\\sheep_label.txt'
# # label2_name = 'E:\\chen\\图卷积——小图标签\\train_label.txt'
#
# f1 = open(label1_name, 'r')
# f2 = open(label2_name, 'r')
#
# for line in f1.readlines():
#     i = line.split()[1]
#     label1.append(int(i) + 1)
# for line in f2.readlines():
#     i = line.split()[1]
#     label2.append(int(i) + 1)
#
# label1 = np.array(label1)
# label1 = label1 / sum(label1)
# label1 = list(label1)
#
# label2 = np.array(label2)
# label2 = label2 / sum(label2)
# label2 = list(label2)
#
# label_1_log = []
# label_2_log = []
# for i in label1:
#     i = math.log(i)
#     label_1_log.append(i)
# for i in label2:
#     i = math.log(i)
#     label_2_log.append(i)
#
# l = get_dist(label_1_log, label_2_log)
#
# print(l)
#
#
# f1.close()
# f2.close()
#
#
