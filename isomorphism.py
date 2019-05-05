import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import math


class transformation_eye():
    array = ([[], []])

    def __init__(self, array):
        self.array = array

    def exchange_two_rows(self, x, y):
        a = self.array[x - 1:x].copy()
        self.array[x - 1:x] = self.array[y - 1:y]
        self.array[y - 1:y] = a
        return self.array

    def exchange_two_cols(self, x, y):
        a = self.array[:, x - 1:x].copy()
        self.array[:, x - 1:x] = self.array[:, y - 1:y]
        self.array[:, y - 1:y] = a
        return self.array

def print_edgelist_to_file(path, graph):
    f = open(path, 'w')
    for node in graph.node:
        for i in graph.adj[node]:
            print(node, ' ', i, file=f)
    f.close()

def tran_graph(sourceG, outpath, counts=20):    # 生成自同构图
    A = nx.to_numpy_matrix(sourceG)
    # A = torch.Tensor(A)
    A = transformation_eye(A)
    l = len(sourceG.nodes)
    for index in range(counts):
        # Al = np.eye(62)
        x = random.randint(1, l)
        y = random.randint(1, l)
        A.exchange_two_rows(x, y)
        A.exchange_two_cols(x, y)
        # Al = exchange_two_rows(Al, x, y)
        # Al = torch.from_numpy(Al)
        # Al = Al.type(torch.FloatTensor)
        # A = torch.mm(Al, A)
        # A = torch.mm(A, Al)
    A = A.array
    A.astype(np.int32)
    G = nx.from_numpy_array(A)
    print_edgelist_to_file(outpath, G)
    return G

label_path = '/home/lmx/chen/graphlets/all_label/'
new_label_path = '/home/lmx/chen/graphlets/new_label/'


def data_enhance(source_path, target_path, addnums=10, count=300):
    H = nx.read_edgelist(source_path)
    p = source_path.split('/')[-1]
    for i in range(addnums):
        f = p.split('.')[0] + '_label.txt'
        label = open(label_path + f, 'r')
        new_label = open(new_label_path + str(i) + 'is' + f, 'w')
        ls = label.readlines()
        for line in ls:
            print(line, file=new_label, end='')
        label.close()
        new_label.close()
        name = target_path + str(i) + 'is' + p
        # print(name)
        tran_graph(H, name, count)
        print(source_path + ' generate one isomorphism graph')

def get_file_name(dir):
    list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            list.append(os.path.join(root, file))
    return list


target_path = '/home/lmx/chen/graphlets/isomorphism/'
filenames = get_file_name('/home/lmx/chen/graphlets/trainset')




for filename in filenames:
    
    data_enhance(filename, target_path, count=3000)