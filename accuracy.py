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
import string



f_truth = open('knn_label_truth_huge_total_mod.txt', 'r')
f_prediction = open('knn_prediction_huge_total_mod.txt', 'r')

lines1 = f_truth.readlines()

lines2 = f_prediction.readlines()

dict_truth = dict()
dict_prediction = dict()

k_range = 20

k = [0 for i in range(k_range)]

total = 122

for i in range(len(lines1)):
    line1_list = lines1[i].split()
    line2_list = lines2[i].split()
    assert(line1_list[0] == line2_list[0])
    dict_truth[line1_list[0]] = []
    dict_prediction[line2_list[0]] = []
    for j in range(k_range):
        dict_truth[line1_list[0]].append(line1_list[j+1].strip(string.digits))
        dict_prediction[line2_list[0]].append(line2_list[j+1].strip(string.digits))
        # dict_truth[line1_list[0]].append(line1_list[j+1])
        # dict_prediction[line2_list[0]].append(line2_list[j+1])
    
        if dict_truth[line1_list[0]][0] in dict_prediction[line2_list[0]][0:j+1]:
            k[j]+=1


# print(k1, k2, k3, k4, k5, k6, k7)

for index, i in enumerate(k):
    print(index+1, float(i)/total)

# print(float(k1)/total, float(k2)/total, float(k3)/total, float(k4)/total, float(k5)/total, float(k6)/total, float(k7)/total)
