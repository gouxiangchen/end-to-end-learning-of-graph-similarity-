import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import math


def print_edgelist_to_file(path, graph):
    f = open(path, 'w')
    for node in graph.node:
        for i in graph.adj[node]:
            print(node, ' ', i, file=f)
    f.close()


def get_file_name(dir):
    list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            list.append(os.path.join(root, file))
    return list


filenames = get_file_name('/home/lmx/chen/graphlets/trainset')
store_path = '/home/lmx/chen/graphlets/noisy'
noisy_count = 10
for filename in filenames:
    g = nx.read_edgelist(filename)
    f = filename.split('/')[-1]
    for k in range(noisy_count):
        
        change_count = 5000
        for i in range(change_count):
            

            is_remove = False
            is_add = False

            while not is_remove and not is_add:
                node1 = random.sample(g.nodes, 10)
                node2 = random.sample(g.nodes, 10)
                for first in node1:
                    for second in node2:
                        if g.has_edge(first, second):
                            if not is_remove:
                                g.remove_edge(first, second)
                                is_remove = True
                        else:
                            if not is_add:
                                g.add_edge(first, second)
                                is_add = True
                        if is_add and is_remove:
                            break
                    if is_add and is_remove:
                        break
                    
            
            # input(len(g.edges))

            # if g.has_edge(node1, node2):
            #     g.remove_edge(node1, node2)
            # else:
            #     g.add_edge(node1, node2)
        print(filename + ' generate one noisy graph')
        ff = store_path + '/' + str(k) + 'no' + f
        print_edgelist_to_file(ff, g)
    # print(f)








# g1 = nx.Graph()
# g = nx.read_edgelist('E:\\chen\\图卷积——小图\\beach.txt')
# store_path = 'E:\\chen\\小图噪声\\beach_noisy_5.txt'
#

