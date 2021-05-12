import sys, os, ctypes
import time
import torch
import argparse
import numpy as np
import random

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')

import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import context.store as store
import _thread
import threading

# example2=ctypes.CDLL('../../cmake/build/example2.cpython-36m-x86_64-linux-gnu.so')

from cmake.build.example2 import *
from context import context

from dist_gcn.models import GCN
import autograd.autograd as atg


def accessG_backProp(needed_G_map_from_workers, i, layerId, epoch):
    if not context.glContext.config['ifBackPropCompress']:
        needed_G_map_from_workers[i] = \
            context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G(
                context.glContext.config['firstHopForWorkers'][i], layerId)
    else:
        needed_G_map_from_workers[i] = \
            context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G_compress(
                context.glContext.config['firstHopForWorkers'][i],
                context.glContext.config['ifBackPropCompensate'], layerId, epoch,
                context.glContext.config['bucketNum_backProp'],
                context.glContext.config['bitNum_backProp']
            )
    store.threadCountList_backProp[layerId - 2] += 1


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}  # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def pullNeighborG(nodes, epoch, layerId):
    # 这个函数主要是补全atg.G2
    # 去指定worker获取一阶邻居
    needed_G_map_from_workers = {}

    start = time.time()
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.worker_id:
            try:
                # t=threading.Thread(target=accessG_backProp,args=(needed_G_map_from_workers, i, layerId, epoch,))
                # t.start()
                # _thread.start_new_thread(accessG_backProp, (needed_G_map_from_workers, i, layerId, epoch))
                accessG_backProp(needed_G_map_from_workers, i, layerId, epoch)
            except:
                print("Error:无法启动线程")
    # layerId start from 2; because the first layer is not need to pass
    # while store.threadCountList_backProp[layerId - 2] != context.glContext.config['worker_num'] - 1:
    #     pass
    # store.threadCountList_backProp[layerId - 2] = 0
    end = time.time()
    print("pull G {0} time:{1}".format(layerId,(end-start)))
    # if layerId == 2:
    #     time_tmp = end - start
    #     store.commTime += time_tmp
    #     print("communication time:{0}".format(store.commTime))
    #     store.commTime = 0
    # else:
    #     time_tmp = end - start
    #     store.commTime += time_tmp

    # if not context.glContext.config['ifBackPropCompress']:
    #     # print("back propagation non-compression")
    #     for i in range(context.glContext.config['worker_num']):
    #         if i != context.glContext.worker_id:
    #             needed_G_map_from_workers[i] = \
    #                 context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G(
    #                     context.glContext.config['firstHopForWorkers'][i], layerId)
    # else:
    #     # print("back propagation compression")
    #     for i in range(context.glContext.config['worker_num']):
    #         if i != context.glContext.worker_id:
    #             needed_G_map_from_workers[i] = \
    #                 context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G_compress(
    #                     context.glContext.config['firstHopForWorkers'][i],
    #                     context.glContext.config['ifBackPropCompensate'], layerId, epoch,
    #                     context.glContext.config['bucketNum_backProp'],
    #                     context.glContext.config['bitNum_backProp']
    #                 )

    needed_G = [None] * (len(context.glContext.newToOldMap) - len(nodes))
    # for循环遍历每个从远端获取的特征
    for wid in range(context.glContext.config['worker_num']):
        if wid != context.glContext.worker_id:
            for i, nid in enumerate(context.glContext.config['firstHopForWorkers'][wid]):
                new_id = context.glContext.oldToNewMap[nid] - len(nodes)
                needed_G[new_id] = needed_G_map_from_workers[wid][i]

    # 将needed_embs转化为tensor
    needed_G = np.array(needed_G)
    needed_G = torch.FloatTensor(needed_G)

    if layerId == 2:
        atg.G2 = torch.cat((atg.G2, needed_G), 0)
    elif layerId == 3:
        atg.G3 = torch.cat((atg.G3, needed_G), 0)

    # print(atg.G2)


def printInfo(firstHopSetsForWorkers):
    print("local and remote vertex distribution:")
    print("local worker {0}:{1}".format(context.glContext.config['id'],
                                        len(firstHopSetsForWorkers[context.glContext.config['id']])))
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.config['id']:
            print("worker {0}:{1}".format(i, len(firstHopSetsForWorkers[i])))


def run_gnn(dgnnClient):
    # 从远程获取顶点信息（主要是边缘顶点一阶邻居信息）后，在本地进行传播
    isRandomData = False
    np.random.seed(1)
    random.seed(1)

    # 这里读取的都是按照server统一编号的；
    nodes_from_server = dgnnClient.nodes
    feat_data_dict_from_server = dgnnClient.features
    labels_from_server = dgnnClient.labels
    adj_lists_from_server = dgnnClient.adjs

    dgnnClient.layerNum = context.glContext.config['layerNum']

    # 开始定义图神经网络模型
    data_num = len(nodes_from_server)

    train_num = int(data_num * 0.53)
    val_num = int(data_num * 0.1)
    test_num = data_num - train_num - val_num

    rand_indices = None
    if isRandomData:
        rand_indices = np.random.permutation(data_num)  # 随机索引
    else:
        rand_indices = np.arange(data_num)
    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:]

    feat_dim = context.glContext.config['feature_dim']

    # 构建本地顶点的一阶邻居集合
    firstHopSet = set()
    for neiborSet in adj_lists_from_server:
        for neiborId in adj_lists_from_server[neiborSet]:
            firstHopSet.add(int(neiborId))

    firstHopSetsForWorkers = context.glContext.config['firstHopForWorkers']
    for i in range(context.glContext.config['worker_num']):
        firstHopSetsForWorkers.append(list())
    for id in firstHopSet:
        workerId = id % context.glContext.config['worker_num']
        firstHopSetsForWorkers[workerId].append(id)

    # transform set to numpy ndarray
    start_trans_set2arr = time.time()
    for i in range(len(firstHopSetsForWorkers)):
        firstHopSetsForWorkers[i] = np.array(firstHopSetsForWorkers[i])
    end_trans_set2arr = time.time()
    # print("transform set to ndarray time:{0} s".format(end_trans_set2arr-start_trans_set2arr))

    # 将feature的dict转化成list
    feat_data = []
    id_old2new_map = {}
    id_new2old_map = {}
    count = 0
    for item in feat_data_dict_from_server.keys():
        feat_data.append(feat_data_dict_from_server[item])
        id_old2new_map[item] = count
        id_new2old_map[count] = item
        count += 1

    # 对一阶邻居编码
    for vid in adj_lists_from_server:
        for neibor_id in adj_lists_from_server[vid]:
            if neibor_id not in id_old2new_map.keys():
                id_old2new_map[neibor_id] = count
                id_new2old_map[count] = neibor_id
                count += 1

    context.glContext.newToOldMap = id_new2old_map
    context.glContext.oldToNewMap = id_old2new_map

    # 将顶点按照id_old2new_map转化
    nodes = [id_old2new_map[i] for i in nodes_from_server]

    # 将邻接表按照id_old2new_map转化
    # 将标签按照id_old2new_map转换
    adjs = []
    labels = []
    for new_id in range(len(nodes_from_server)):
        old_id = id_new2old_map[new_id]
        neibor_set_old = adj_lists_from_server[old_id]
        neibor_set_new = []
        # 转化labels
        labels.append(labels_from_server[old_id])

        for item in neibor_set_old:
            neibor_set_new.append(id_old2new_map[item])
        adjs.append(neibor_set_new)

    labels = np.array(labels)

    # feat_data,nodes,adjs,labels重置了顺序
    # 其中feat_data包含了临界的一阶邻居的特征，而nodes里的id就对应了feat_data, adjs, labels对应维度

    # features = nn.Embedding(len(feat_data), feat_dim)  # 构建特征嵌入矩阵
    # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)  # 构建特征值权重矩阵
    features = sp.csr_matrix(feat_data, dtype=np.float32)
    features = normalize(features)
    edges = []
    # 从adj中解析出edge
    for i in range(len(adjs)):
        for nei_id in adjs[i]:
            edges.append([i, nei_id])

    edges = np.array(edges)

    model = GCN(nfeat=feat_dim,
                nhid=context.glContext.config['hidden'],
                nclass=7,
                dropout=0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=8e-4)

    # 将邻接矩阵处理成coo的格式
    # edges=[[0,5],[0,10],[0,100],[2,100],[5,1002]]
    # edges=np.array(edges)
    adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(len(id_old2new_map), len(id_old2new_map)),
                         dtype=np.int)

    # print(adjs.T)

    adjs = adjs + adjs.T.multiply(adjs.T > adjs) - adjs.multiply(adjs.T > adjs)
    adjs = normalize(adjs + sp.eye(adjs.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adjs = adjs[nodes]

    adjs = sparse_mx_to_torch_sparse_tensor(adjs)  # 邻接矩阵转为tensor处理

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    ifCompress = context.glContext.config['ifCompress']

    # error matrix
    errorW0 = None
    errorW1 = None
    errorB0 = None
    errorB1 = None

    errorHA1 = None

    printInfo(firstHopSetsForWorkers)

    for epoch in range(context.glContext.config['IterNum']):
        startTimeTotle = time.time()
        model.train()
        optimizer.zero_grad()
        store.isTrain = True
        if ifCompress:
            context.glContext.config['ifCompress'] = True

        # slow
        output = model(features, adjs, nodes_from_server, epoch)

        if context.glContext.config['isNeededExactBackProp']:
            if context.glContext.config["layerNum"] == 2:
                atg.H2 = output
                atg.Z2.required_grad = True
                atg.Z2.retain_grad()
            elif context.glContext.config["layerNum"] == 3:
                atg.H3 = output
                atg.Z3.required_grad = True
                atg.Z3.retain_grad()

        # print(output)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，
        # 这里就要使用CrossEntropyLoss了
        # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
        # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
        # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
        # https://blog.csdn.net/hao5335156/article/details/80607732
        acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
        loss_train.backward()  # 反向求导  Back Propagation

        # 需要准确的反向传播过程
        if context.glContext.config['isNeededExactBackProp']:
            if context.glContext.config["layerNum"] == 2:
                atg.G2 = atg.Z2.grad
                # 算出G2后，给自己所在的server赋值
                G_list = atg.G2.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]

                dgnnClient.setG(G_dict, 2)
                # 同步一下

                # G2本地完全的，不需要从其他机器获取
                if context.Context.config['isHACompensate']:
                    if epoch == 0:
                        errorHA1 = np.random.random(atg.A_X_H1.shape)
                        atg.A_X_H1 = atg.A_X_H1 - torch.FloatTensor(errorHA1)


                    else:
                        atg.A_X_H1 = atg.A_X_H1 + torch.FloatTensor(errorHA1)
                        errorHA1 = np.random.random(atg.A_X_H1.shape)
                        atg.A_X_H1 = atg.A_X_H1 - torch.FloatTensor(errorHA1)

                atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)
                atg.sigma_z1_grad = atg.Z1.data
                atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))

                # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
                # 这段跟获取一阶邻居的嵌入表示类似,pull g2 to compute g1
                start = time.time()

                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                pullNeighborG(nodes, epoch, 2)
                end = time.time()
                # if context.glContext.ifShowInfo:
                # print("pull neiborgh G time:{0}".format(end - start))

                a = torch.spmm(adjs, atg.G2)
                b = torch.mm(a, model.gc2.weight.t())
                atg.G1 = torch.mul(b, atg.sigma_z1_grad)
                atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)
                atg.B1 = atg.G2.detach().numpy().sum(axis=0)
                atg.B0 = atg.G1.detach().numpy().sum(axis=0)
                np.set_printoptions(threshold=sys.maxsize)
                # set a random error matrix
                if context.Context.config['isGradientComensate']:
                    if epoch == 0:
                        errorW0 = np.random.random(atg.Y0.shape) / 1000
                        errorW1 = np.random.random(atg.Y1.shape) / 1000
                        errorB0 = np.random.random(atg.B0.shape) / 1000
                        errorB1 = np.random.random(atg.B1.shape) / 1000
                        model.gc1.weight.grad.data = atg.Y0 - torch.tensor(errorW0)
                        model.gc1.bias.grad.data = torch.FloatTensor(atg.B0 - errorB0)
                        model.gc2.weight.grad.data = atg.Y1 - torch.tensor(errorW1)
                        model.gc2.bias.grad.data = torch.FloatTensor(atg.B1 - errorB1)
                    else:
                        # atg.Y0 = atg.Y0 + torch.tensor(errorW0)
                        # atg.B0 = atg.B0 + errorB0
                        # atg.Y1 = atg.Y1 + torch.tensor(errorW1)
                        # atg.B1 = atg.B1 + errorB1
                        errorW0 = np.random.random(atg.Y0.shape) / 1000
                        errorW1 = np.random.random(atg.Y1.shape) / 1000
                        errorB0 = np.random.random(atg.B0.shape) / 1000
                        errorB1 = np.random.random(atg.B1.shape) / 1000
                        model.gc1.weight.grad.data = atg.Y0 - torch.tensor(errorW0)
                        model.gc1.bias.grad.data = torch.FloatTensor(atg.B0 - errorB0)
                        model.gc2.weight.grad.data = atg.Y1 - torch.tensor(errorW1)
                        model.gc2.bias.grad.data = torch.FloatTensor(atg.B1 - errorB1)
                else:
                    model.gc1.weight.grad.data = atg.Y0
                    model.gc1.bias.grad.data = torch.FloatTensor(atg.B0)
                    model.gc2.weight.grad.data = atg.Y1
                    model.gc2.bias.grad.data = torch.FloatTensor(atg.B1)

                # print(atg.G1.data_raw)
                # print(atg.G2.data_raw)
                # print('end')

                # print('e{0},l1: weight00:{1}'.format(epoch,atg.Y1[0][0]))

            elif context.glContext.config["layerNum"] == 3:
                atg.G3 = atg.Z3.grad
                # 算出G2后，给自己所在的server赋值
                G_list = atg.G3.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]

                dgnnClient.setG(G_dict, 3)
                # 同步一下

                # G3本地完全的，不需要从其他机器获取
                atg.Y2 = torch.mm(atg.A_X_H2.t(), atg.G3)
                atg.B2 = atg.G3.detach().numpy().sum(axis=0)

                atg.sigma_z2_grad = atg.Z2.data
                atg.sigma_z2_grad = torch.tensor(np.where(atg.sigma_z2_grad > 0, 1, 0))
                # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
                # 这段跟获取一阶邻居的嵌入表示类似
                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                pullNeighborG(nodes, epoch, 3)
                a = torch.spmm(adjs, atg.G3)
                b = torch.mm(a, model.gc3.weight.t())
                atg.G2 = torch.mul(b, atg.sigma_z2_grad)
                atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)
                atg.B1 = atg.G2.detach().numpy().sum(axis=0)

                # set G2
                G_list = atg.G2.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]
                dgnnClient.setG(G_dict, 2)

                atg.sigma_z1_grad = atg.Z1.data
                atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))

                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                pullNeighborG(nodes, epoch, 2)
                a = torch.spmm(adjs, atg.G2)
                b = torch.mm(a, model.gc2.weight.t())
                atg.G1 = torch.mul(b, atg.sigma_z1_grad)
                atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)
                atg.B0 = atg.G1.detach().numpy().sum(axis=0)

                np.set_printoptions(threshold=sys.maxsize)
                model.gc1.weight.grad.data = atg.Y0
                model.gc1.bias.grad.data = torch.FloatTensor(atg.B0)
                model.gc2.weight.grad.data = atg.Y1
                model.gc2.bias.grad.data = torch.FloatTensor(atg.B1)
                model.gc3.weight.grad.data = atg.Y2
                model.gc3.bias.grad.data = torch.FloatTensor(atg.B2)

        # 求出梯度后，发送到参数服务器中进行聚合，并更新参数值
        # a=model.gc1.weight.grad
        # 将权重梯度和偏移梯度分别转化成map<int,vector<vector<float>>>和map<int,vector<float>>
        weights_grad_map = {}
        bias_grad_map = {}

        # 这里先不做通用的，直接假定是两层，之后再改
        if context.glContext.config["layerNum"] == 2:
            weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
            weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
            bias_grad_map[0] = model.gc1.bias.grad.detach().numpy().tolist()
            bias_grad_map[1] = model.gc2.bias.grad.detach().numpy().tolist()
        elif context.glContext.config["layerNum"] == 3:
            weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
            weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
            weights_grad_map[2] = model.gc3.weight.grad.detach().numpy().tolist()
            bias_grad_map[0] = model.gc1.bias.grad.detach().numpy().tolist()
            bias_grad_map[1] = model.gc2.bias.grad.detach().numpy().tolist()
            bias_grad_map[2] = model.gc3.bias.grad.detach().numpy().tolist()
        # print(weights_grad_map[0])
        # print(weights_grad_map[1])
        # print(bias_grad_map[0])
        # print(bias_grad_map[1])

        start = time.time()
        dimList = []
        dimList.append(context.glContext.config['feature_dim'])
        for i in range(context.glContext.config['layerNum'] - 1):
            dimList.append(context.glContext.config['hidden'][i])

        server_num = context.glContext.config['server_num']
        for i in range(server_num):
            weights_grad_map_each = {}
            for j in range(context.glContext.config['layerNum']):
                eachNum = int(dimList[j] / server_num)
                if i == server_num - 1:
                    weights_grad_map_each[j] = weights_grad_map[j][(server_num - 1) * eachNum:dimList[j]]
                else:
                    weights_grad_map_each[j] = weights_grad_map[j][i * eachNum:(i + 1) * eachNum]

            if i == 0:
                context.glContext.dgnnServerRouter[0].sendAndUpdateModels(context.glContext.config['id'], i,
                                                                          weights_grad_map_each,
                                                                          bias_grad_map)
            else:
                context.glContext.dgnnServerRouter[i].sendAndUpdateModels(context.glContext.config['id'], i,
                                                                          weights_grad_map_each,
                                                                          bias_grad_map)

        end = time.time()

        print("update model time:{0}".format(end - start))
        endTimeTotle = time.time()


        # 返回参数后同步,这里barrier的参数暂时没有用到
        context.glContext.dgnnServerRouter[0].server_Barrier(0)

        # optimizer.step()  # 更新所有的参数  Gradient Descent
        model.eval()
        # output = model(features, adjs,nodes_from_server)



        if epoch % 1 == 0:
            # context.glContext.config['ifCompress'] = False
            # store.isTrain = False
            # output = model(features, adjs, nodes_from_server, 10000)
            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 验证集的损失函数
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
            # acc_entire = context.glContext.dgnnServerRouter[0].sendAccuracy(acc_val, acc_train)
            # print('Epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'acc_train: {:.4f}'.format(acc_train.item()),
            #       "iter_time: {0}".format(endTimeTotle - startTimeTotle))
            #
            # print('acc_train_entire{:.4f}'.format(acc_entire['train']),
            #       'acc_val_entire:{:.4f}'.format(acc_entire['val']))
            #
            print('Epoch: {:04d}'.format(epoch + 1),
                  "iter_time: {0}".format(endTimeTotle - startTimeTotle))

    context.glContext.config['ifCompress'] = False
    store.isTrain = False
    output = model(features, adjs, nodes_from_server, 10000)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])  # 验证集的损失函数
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()))


def parserInit():
    parser = argparse.ArgumentParser(description="Pytorch argument parser")
    parser.add_argument('--role', type=str, help='machine role')
    parser.add_argument('--id', type=int, help='the id of role')
    parser.add_argument('--mode', type=str, help='mode')
    parser.add_argument('--worker_num', type=int, help='the number of worker')
    parser.add_argument('--server_num', type=int, help='the number of server')
    parser.add_argument('--data_num', type=int, help='the number of data_raw')
    parser.add_argument('--feature_dim', type=int, help='the dim size of feature')
    parser.add_argument('--class_num', type=int, help='the number of data_raw')
    parser.add_argument('--hidden', type=str, help='hidden')
    parser.add_argument('--ifCompress', type=str, help='ifCompress')
    parser.add_argument('--ifCompensate', type=str, help='ifCompensate')
    parser.add_argument('--data_path', type=str, help='data_path')

    parser.add_argument('--isNeededExactBackProp', type=str, help='isNeededExactBackProp')
    parser.add_argument('--bucketNum', type=int, help='bucketNum')
    parser.add_argument('--IterNum', type=int, help='IterNum')
    parser.add_argument('--ifBackPropCompress', type=str, help='ifBackPropCompress')
    parser.add_argument('--ifBackPropCompensate', type=str, help='ifBackPropCompensate')

    parser.add_argument('--bucketNum_backProp', type=int, help='bucketNum_backProp')
    parser.add_argument('--changeToIter', type=int, help='changeToIter')
    parser.add_argument('--compensateMethod', type=str, help='compensateMethod')
    parser.add_argument('--isChangeRate', type=str, help='isChangeRate')
    parser.add_argument('--bitNum', type=int, help='bitNum')
    parser.add_argument('--trend', type=int, help='trend')
    parser.add_argument('--bitNum_backProp', type=int, help='bitNum_backProp')
    parser.add_argument('--localCodeMode', type=str, help='localCodeMode')

    args = parser.parse_args()
    if args.localCodeMode == 'true':
        context.Context.localCodeMode = True
    else:
        context.Context.localCodeMode = False

    if context.Context.localCodeMode:
        print("setting mode as code")
        context.glContext.config['role'] = args.role
        context.glContext.config['id'] = args.id
        context.glContext.config['mode'] = 'code'
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1
    else:
        print("setting mode as test")
        context.glContext.config['role'] = args.role
        context.glContext.config['id'] = args.id
        context.glContext.config['mode'] = args.mode
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1
        context.glContext.config['worker_num'] = args.worker_num
        context.glContext.config['server_num'] = args.server_num
        context.glContext.config['data_num'] = args.data_num
        context.glContext.config['feature_dim'] = args.feature_dim
        context.glContext.config['class_num'] = args.class_num
        context.glContext.config['hidden'] = list(map(int, args.hidden.split(',')))
        context.glContext.config['ifCompensate'] = args.ifCompensate

        context.glContext.config['isNeededExactBackProp'] = args.isNeededExactBackProp
        context.glContext.config['bucketNum'] = args.bucketNum
        context.glContext.config['IterNum'] = args.IterNum
        context.glContext.config['ifBackPropCompress'] = args.ifBackPropCompress
        context.glContext.config['ifBackPropCompensate'] = args.ifBackPropCompensate

        context.glContext.config['bucketNum_backProp'] = args.bucketNum_backProp
        context.glContext.config['changeToIter'] = args.changeToIter
        context.glContext.config['compensateMethod'] = args.compensateMethod
        context.glContext.config['data_path'] = args.data_path
        context.glContext.config['bitNum'] = args.bitNum
        context.glContext.config['trend'] = args.trend
        context.glContext.config['bitNum_backProp'] = args.bitNum_backProp

        if args.isChangeRate == 'false':
            context.glContext.config['isChangeRate'] = False
        else:
            context.glContext.config['isChangeRate'] = True

        if args.ifCompress == 'false':
            context.glContext.config['ifCompress'] = False
        elif args.ifCompress == 'true':
            context.glContext.config['ifCompress'] = True

        if args.ifCompensate == 'false':
            context.glContext.config['ifCompensate'] = False
        elif args.ifCompensate == 'true':
            context.glContext.config['ifCompensate'] = True

        if args.isNeededExactBackProp == 'false':
            context.glContext.config['isNeededExactBackProp'] = False
        elif args.isNeededExactBackProp == 'true':
            context.glContext.config['isNeededExactBackProp'] = True

        if args.ifBackPropCompress == 'false':
            context.glContext.config['ifBackPropCompress'] = False
        elif args.ifBackPropCompress == 'true':
            context.glContext.config['ifBackPropCompress'] = True

        if args.ifBackPropCompensate == 'false':
            context.glContext.config['ifBackPropCompensate'] = False
        elif args.ifBackPropCompensate == 'true':
            context.glContext.config['ifBackPropCompensate'] = True

    context.glContext.ipInit()
    store.init()
    print('server:{0}'.format(context.glContext.config['server_address']))
    print('master:{0}'.format(context.glContext.config['master_address']))
    print('worker:{0}'.format(context.glContext.config['worker_address']))


def printContext():
    print("role={0},id={1},mode={2},worker_num={3},data_num={4},isNeededExactBackProp={5},feature_dim={6},"
          "class_num={7},hidden={8},ifCompress={9},ifCompensate={10},bucketNum={11},IterNum={12},ifBackPropCompress={13},"
          "ifBackPropCompensate={14},bucketNum_backProp={15},changeToIter={16},compensateMethod={17}"
          .format(context.glContext.config['role'], context.glContext.config['id'], context.glContext.config['mode'],
                  context.glContext.config['worker_num'],
                  context.glContext.config['data_num'], context.glContext.config['isNeededExactBackProp'],
                  context.glContext.config['feature_dim'],
                  context.glContext.config['class_num'], context.glContext.config['hidden'],
                  context.glContext.config['ifCompress'],
                  context.glContext.config['ifCompensate'], context.glContext.config['bucketNum'],
                  context.glContext.config['IterNum'], context.glContext.config['ifBackPropCompress'],
                  context.glContext.config['ifBackPropCompensate'],
                  context.glContext.config['bucketNum_backProp'], context.glContext.config['changeToIter'],
                  context.glContext.config['compensateMethod']))


if __name__ == "__main__":
    parserInit()
    printContext()
    # context.glContext.config['data_num']=args.datanum
    # context.glContext.config['data_num']=args.datanum

    if context.glContext.config['role'] == 'server':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['server_address'][context.glContext.config['id']],
                                  context.glContext.worker_id)
        # ServiceImpl.RunServerByPy("127.0.0.1:2001")

    elif context.glContext.config['role'] == 'worker':
        context.glContext.dgnnServerRouter = []
        context.glContext.dgnnWorkerRouter = []
        context.glContext.dgnnClient = DGNNClient()
        context.glContext.dgnnMasterRouter = DGNNClient()
        context.glContext.dgnnClientRouterForCpp = Router()

        context.glContext.worker_id = context.glContext.config['id']
        id = context.glContext.config['id']
        # 当前机器的客户端，需要启动server，以保证不同机器间中间表征向量传输

        context.glContext.dgnnClient.serverAddress = context.glContext.config['worker_address'][id]

        context.glContext.dgnnClient.startClientServer()
        for i in range(context.glContext.config['server_num']):
            context.glContext.dgnnServerRouter.insert(i, DGNNClient())
            context.glContext.dgnnServerRouter[i].init_by_address(context.glContext.config['server_address'][i])
        for i in range(context.glContext.config['worker_num']):
            context.glContext.dgnnWorkerRouter.insert(i, DGNNClient())
            context.glContext.dgnnWorkerRouter[i].init_by_address(context.glContext.config['worker_address'][i])

        context.glContext.dgnnMasterRouter.init_by_address(context.glContext.config['master_address'])

        # 在c++端初始化dgnnWorkerRouter
        context.glContext.dgnnClientRouterForCpp.initWorkerRouter(context.glContext.config['worker_address'])

        # 所有创建的类都在一个进程里，通过c++对静态变量操作，在所有类中都可见
        # print(dgnnClient.testString)
        # print(dgnnMasterRouter.testString)
        # print(dgnnServerRouter[0].testString)

        # 从master中获取各自需要的数据,这里默认启动了hash划分
        context.glContext.dgnnMasterRouter.pullDataFromMaster(
            id, context.glContext.config['worker_num'],
            context.glContext.config['data_num'],
            context.glContext.config['data_path'],
            context.glContext.config['feature_dim'],
            context.glContext.config['class_num'])

        # 初始化参数服务器模型，现在假设参数服务器就一台机器，先不进行参数划分
        # 输入：节点属性维度、隐藏层维度、标签维度

        for i in range(context.glContext.config['server_num']):
            context.glContext.dgnnServerRouter[i].initParameter(
                context.glContext.config['worker_num'],
                context.glContext.config['server_num'],
                context.glContext.config['feature_dim'],
                context.glContext.config['hidden'],
                context.glContext.config['class_num'],
                id
            )

        # # 已经将各自数据放到了数据库里，接下来定义GNN模型，然后训练
        # # 这里的操作的workerStore是全局静态变量，因此整个进程都可见
        run_gnn(context.glContext.dgnnClient)

    elif context.glContext.config['role'] == 'master':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)
        # ServiceImpl.RunServerByPy('127.0.0.1:4001')
