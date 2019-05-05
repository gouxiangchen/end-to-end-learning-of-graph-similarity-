import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import math


def exchange_two_rows(matirx, x, y):    # 只适用单位矩阵
    matirx[x - 1][x - 1] = 0
    matirx[x - 1][y - 1] = 1
    matirx[y - 1][y - 1] = 0
    matirx[y - 1][x - 1] = 1
    return matirx


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



def process_txt(inpath, outpath, mode):
    if mode == 0:
        fin = open(inpath, 'r')
        fout = open(outpath, 'w')
        for line in fin:
            words = line.split('\t')
            fout.write(words[0] + '\t' + words[1] + '\n')
        fin.close()
        fout.close()
    elif mode == 1:
        fin = open(inpath, 'r')
        fout = open(outpath, 'w')
        for line in fin:
            words = line.split()
            fout.write(words[0] + '\t' + words[1] + '\n')
        fin.close()
        fout.close()
    else:
        pass


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


def data_enhance(source_path, addnums=10, count=300):
    H = nx.read_edgelist(source_path)
    p = source_path.split('.')
    for i in range(addnums):
        name = p[0] + '_' + str(i) + '.txt'
        tran_graph(H, name, count)


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
        r = torch.Tensor().cuda()
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
        self.gc1 = myLayer(128, 64).cuda()
        self.gc2 = myLayer(64, 32).cuda()
        self.gc3 = myLayer(32, 16).cuda()
        self.ReLU = nn.ReLU().cuda()
        self.combine = combineLayer(16, 16).cuda()
        self.fc1 = nn.Linear(16, 8).cuda()
        self.fc2 = nn.Linear(8, 4).cuda()
        self.fc3 = nn.Linear(4, 1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

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
        X_po = torch.Tensor([1 for i in range(Nx)]).cuda()
        Y_po = torch.Tensor([1 for i in range(Ny)]).cuda()
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

# H = nx.read_edgelist('dataset_small\\powergrid.txt')
# G = tran_graph(H, 'dataset_small\\powergrid_1.txt', 30)
# data_enhance('dataset_small\\powergrid_1.txt')


def get_file_name(dir):
    list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            list.append(os.path.join(root, file))
    return list


def str2int_evalue(numstr):
    l = numstr.split('e')
    if(len(l) == 1):
        return float(numstr)
    else:
        p = int(l[1])
        d = float(l[0])
        r = d * math.pow(10, p)
        return r


def get_dist(l1, l2):
    r = 0.
    for i in range(len(l1)):
        k = (l1[i] - l2[i]) * (l1[i] - l2[i])
        r += k
    r = math.sqrt(r)
    return r

mynet = graphConvNet()
mynet = mynet.cuda()
# G1 = nx.read_edgelist('dataset_small\pdzbase_1.txt')
# G2 = nx.read_edgelist('dataset_small\pdzbase_1_2.txt')
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
# z = mynet(A1, D1, H1, A2, D2, H2)
# print(z)

label_names = get_file_name('/home/lmx/chen/graphlets/all_label')
label_dict = dict()
for label_name in label_names:
    # print('label name:', label_name.split('_')[2].split('\\')[1])
    graphlets = []
    f = open(label_name, 'r')
    for line in f.readlines():
        i = line.split()[1]
        graphlets.append(int(i)+1)
    f.close()
    frequency = np.array(graphlets)
    frequency = frequency / sum(frequency)
    frequency = list(frequency)
    label_dict[label_name.split('/')[-1].split('.')[0].split('_')[0]] = frequency

label_name = []
for key in label_dict:
    graphlets = []
    for i in label_dict[key]:
        i = math.log(i)
        graphlets.append(i)
    label_dict[key] = graphlets
    label_name.append(key)



# for index, key in enumerate(label_dict):
#     print(index, key, (label_dict[key]))

# print(np.array(label_dict['ChicagoRegional']), np.array(label_dict['euroroad']))
# print(math.exp(-get_dist(label_dict['ChicagoRegional'], label_dict['euroroad'])))

# filenames = get_file_name('D:\pycharmProject\\venv\dataset_small')
# filenames2 = filenames.copy()
# random.shuffle(filenames)
# random.shuffle(filenames2)
# for filename_x in filenames:
#     print(filename_x.split('_')[1].split('\\')[1])
#


start = time.clock()

optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
loss_func = nn.MSELoss()
EPOCH = 128
model_name = 'huge_dataset_ruin3_'
filenames = get_file_name('/home/lmx/chen/graphlets/trainset')
model_counts = 0


f = open('loss_huge_dataset.txt', 'a+', encoding='utf-8')

mynet.load_state_dict(torch.load('/home/lmx/chen/graphlets/model/huge_dataset_ruin2_39epoch.para'))

print('train size : ', len(filenames))

for epoch in range(EPOCH):
    step = 0
    if epoch % 3 == 0:
        torch.save(mynet.state_dict(), '/home/lmx/chen/graphlets/model/' + model_name + str(epoch) + 'epoch.para')
    filenames2 = filenames.copy()
    filenames = random.sample(filenames, 200)
    filenames2 = random.sample(filenames2, 200)
    for filename_x in filenames:
        co = 0
        loss = 0
        for filename_y in filenames2:
            x = filename_x.split('/')[-1].split('.')[0]
            y = filename_y.split('/')[-1].split('.')[0]
            # l = math.exp(- (get_dist(label_dict[x], label_dict[y]))/10)
            l = get_dist(label_dict[x], label_dict[y])
            label = torch.Tensor([l]).cuda()
            label.squeeze()

            G1 = nx.read_edgelist(filename_x)
            G2 = nx.read_edgelist(filename_y)
            A1 = torch.Tensor(nx.to_numpy_matrix(G1)).cuda()
            A2 = torch.Tensor(nx.to_numpy_matrix(G2)).cuda()
            H1 = torch.eye(A1.size(0), 128).cuda()
            H2 = torch.eye(A2.size(0), 128).cuda()
            list1 = []
            list2 = []
            for node in G1.nodes:
                list1.append(G1.degree(node))
            for node in G2.nodes:
                list2.append(G2.degree(node))
            D1 = torch.diag(torch.Tensor(list1)).cuda()
            D2 = torch.diag(torch.Tensor(list2)).cuda()
            out = mynet(A1, D1, H1, A2, D2, H2)
            
            loss += loss_func(out, label)
            co += 1

            if co % 50 == 0:
                print('||out: ', out, '|| label: ', label)
                print(filename_x, filename_y, 'see 50')

            
            # print('epoch: ', epoch, '||step : ', step, ' ||loss: ', loss)
            del G1
            del G2
            del A1
            del A2
            del H1
            del H2
            del D1
            del D2
            del out
            del label
        torch.cuda.empty_cache()
        loss /= co
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('epoch: ', epoch, '||step : ', step, ' ||loss: ', loss)
        print(filename_x, 'epoch: ', epoch, '||step : ', step, ' ||loss: ', loss, file=f)

        step += 1
        loss = loss.cpu()
        if float(loss.data.numpy()) < 0.09 and model_counts < 10:
            torch.save(mynet.state_dict(), model_name + str(model_counts) + 'mix_random_low.para')
        elif float(loss.data.numpy()) < 0.01 and model_counts < 30:
            torch.save(mynet.state_dict(), model_name + str(model_counts) + 'mix_random_low.para')


torch.save(mynet.state_dict(), 'huge_dataset_model_total.para')
end = time.clock()

print(str(end-start), file=f)
f.close()
