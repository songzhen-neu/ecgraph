import torch
import numpy as np
import scipy.sparse as sp
from cmake.build.lib.pb11_ec import *
from ecgraph.context import context
import random
import time
import psutil,os
from ecgraph.data_structure.graph import Graph


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


def normalize_gcn(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -0.5).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float)  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def normalize_feature(mx):
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


# Yu
def load_data(dgnnClient):
    isRandomData = False
    # np.random.seed(7)
    # random.seed(7)
    # global ids encoded by master
    start=time.time()
    print("memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    nodes_from_server = dgnnClient.nodes
    feats_from_server = dgnnClient.features
    labels_from_server = dgnnClient.labels
    adjs_from_server = dgnnClient.adjs
    end=time.time()
    print("time:{:.4f}".format(end-start))
    print("memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    worker_contains_nodes = dgnnClient.nodesForEachWorker
    nodes_in_worker = {}

    # dgnnClient.freeSpace()

    # build
    for i in range(context.glContext.config['worker_num']):
        for j in range(len(worker_contains_nodes[i])):
            nodes_in_worker[worker_contains_nodes[i][j]] = i

    dgnnClient.layerNum = context.glContext.config['layerNum']
    data_num = len(nodes_from_server)

    train_ratio = context.glContext.config['train_num'] / context.glContext.config['data_num']
    val_ratio = context.glContext.config['val_num'] / context.glContext.config['data_num']
    test_ratio = context.glContext.config['test_num'] / context.glContext.config['data_num']

    train_num = int(data_num * train_ratio)
    val_num = int(data_num * val_ratio)
    test_num = int(data_num * test_ratio)

    if isRandomData:
        rand_indices = np.random.permutation(data_num)  # 随机索引
    else:
        rand_indices = np.arange(data_num)
    # idx_train 本地full graph newly encoding
    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    idx_train_old = [nodes_from_server[i] for i in idx_train]

    idx_train_for_test=idx_train

    # build the first-hop neighboring set (containing the local and remote neighbors)
    first_hop_set = set()
    for neighbor_set in adjs_from_server:
        for neighbor_id in adjs_from_server[neighbor_set]:
            first_hop_set.add(int(neighbor_id))

    # first_hop_set_for_workers = context.glContext.config['firstHopForWorkers']
    first_hop_set_for_workers = []
    for i in range(context.glContext.config['worker_num']):
        first_hop_set_for_workers.append(list())
    for id in first_hop_set:
        workerId = nodes_in_worker[id]
        first_hop_set_for_workers[workerId].append(id)

    # transform set to numpy ndarray
    for i in range(len(first_hop_set_for_workers)):
        first_hop_set_for_workers[i] = np.array(first_hop_set_for_workers[i])

    # 将feature的dict转化成list
    feat_data = []
    id_old2new_map = {}
    id_new2old_map = {}
    count = 0
    for item in feats_from_server.keys():
        feat_data.append(feats_from_server[item])
        id_old2new_map[item] = count
        id_new2old_map[count] = item
        count += 1

    # 对一阶邻居编码
    for vid in adjs_from_server:
        for neibor_id in adjs_from_server[vid]:
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
    for new_id in range(data_num):
        old_id = id_new2old_map[new_id]
        neibor_set_old = adjs_from_server[old_id]
        neibor_set_new = []
        # 转化labels
        labels.append(labels_from_server[old_id])

        for item in neibor_set_old:
            neibor_set_new.append(id_old2new_map[item])
        adjs.append(neibor_set_new)

    labels = np.array(labels)

    feat_data = sp.csr_matrix(feat_data, dtype=np.float32)

    feat_data = torch.FloatTensor(np.array(feat_data.todense()))
    labels = torch.LongTensor(labels)

    graph_full = Graph(nodes_from_server, feat_data, labels, adjs, id_old2new_map, id_new2old_map,
                       first_hop_set_for_workers)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return {'idx_val': idx_val,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'idx_train_for_test':idx_train,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'train_num': train_num,
            'test_num': test_num,
            'val_num': val_num,
            'graph_full': graph_full,}
            # 'graph_train': graph_train}



def load_data_sample(dgnnClient):
    isRandomData = False
    # np.random.seed(2)
    # random.seed(2)
    # 这里读取的都是按照master统一编号的,
    nodes_from_server = dgnnClient.nodes
    feats_from_server = dgnnClient.features
    labels_from_server = dgnnClient.labels
    adjs_from_server = dgnnClient.adjs

    worker_contains_nodes = dgnnClient.nodesForEachWorker
    nodes_in_worker = {}

    # dgnnClient.freeSpace()

    # build
    for i in range(context.glContext.config['worker_num']):
        for j in range(len(worker_contains_nodes[i])):
            nodes_in_worker[worker_contains_nodes[i][j]] = i

    dgnnClient.layerNum = context.glContext.config['layerNum']
    data_num = len(nodes_from_server)

    train_ratio = context.glContext.config['train_num'] / context.glContext.config['data_num']
    val_ratio = context.glContext.config['val_num'] / context.glContext.config['data_num']
    test_ratio = context.glContext.config['test_num'] / context.glContext.config['data_num']

    train_num = int(data_num * train_ratio)
    val_num = int(data_num * val_ratio)
    test_num = int(data_num * test_ratio)

    if isRandomData:
        rand_indices = np.random.permutation(data_num)  # 随机索引
    else:
        rand_indices = np.arange(data_num)
    # idx_train 本地full graph newly encoding
    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    idx_train_old = [nodes_from_server[i] for i in idx_train]


    # build the first-hop neighboring set (containing the local and remote neighbors)
    first_hop_set = set()
    for neighbor_set in adjs_from_server:
        for neighbor_id in adjs_from_server[neighbor_set]:
            first_hop_set.add(int(neighbor_id))

    # first_hop_set_for_workers = context.glContext.config['firstHopForWorkers']
    first_hop_set_for_workers = []
    for i in range(context.glContext.config['worker_num']):
        first_hop_set_for_workers.append(list())
    for id in first_hop_set:
        workerId = nodes_in_worker[id]
        first_hop_set_for_workers[workerId].append(id)

    # transform set to numpy ndarray
    for i in range(len(first_hop_set_for_workers)):
        first_hop_set_for_workers[i] = np.array(first_hop_set_for_workers[i])

    # 将feature的dict转化成list
    feat_data = []
    id_old2new_map = {}
    id_new2old_map = {}
    count = 0
    for item in feats_from_server.keys():
        feat_data.append(feats_from_server[item])
        id_old2new_map[item] = count
        id_new2old_map[count] = item
        count += 1

    # 对一阶邻居编码
    for vid in adjs_from_server:
        for neibor_id in adjs_from_server[vid]:
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
    for new_id in range(data_num):
        old_id = id_new2old_map[new_id]
        neibor_set_old = adjs_from_server[old_id]
        neibor_set_new = []
        # 转化labels
        labels.append(labels_from_server[old_id])

        for item in neibor_set_old:
            neibor_set_new.append(id_old2new_map[item])
        adjs.append(neibor_set_new)

    labels = np.array(labels)

    feat_data = sp.csr_matrix(feat_data, dtype=np.float32)

    feat_data = torch.FloatTensor(np.array(feat_data.todense()))
    labels = torch.LongTensor(labels)

    graph_full = Graph(nodes_from_server, feat_data, labels, adjs, id_old2new_map, id_new2old_map,
                       first_hop_set_for_workers)


    # sample the neighbors
    for key in adjs_from_server:
        if len(adjs_from_server[key])<= context.glContext.config['neigh_sam'][0]:
            pass
        else:
            adjs_from_server[key]=set(list(adjs_from_server[key])[0:context.glContext.config['neigh_sam'][0]])

    all_train_node_set, train_local = all_agg_node_num_all(nodes_from_server, idx_train, adjs_from_server)
    train_node_old_id = train_local
    train_node_old_id = sorted(train_node_old_id)


    first_hop_set_train = set()
    # idx_train:
    for vid in train_node_old_id:
        for nid in adjs_from_server[vid]:
            if nid in all_train_node_set:
                first_hop_set_train.add(int(nid))

    first_hop_set_for_workers_train = []
    for i in range(context.glContext.config['worker_num']):
        first_hop_set_for_workers_train.append(list())


    for id in first_hop_set_train:
        workerId = nodes_in_worker[id]
        first_hop_set_for_workers_train[workerId].append(id)



    # transform set to numpy ndarray
    for i in range(len(first_hop_set_for_workers_train)):
        first_hop_set_for_workers_train[i] = np.array(sorted(first_hop_set_for_workers_train[i]))


    # 将feature的dict转化成list
    feat_data_train = []
    id_old2new_map_train = {}
    id_new2old_map_train = {}
    count_train = 0
    for item in train_node_old_id:
        feat_data_train.append(feats_from_server[item])
        id_old2new_map_train[item] = count_train
        id_new2old_map_train[count_train] = item
        count_train += 1

    # 对一阶邻居编码
    for vid in train_node_old_id:
        for neibor_id in adjs_from_server[vid]:
            if neibor_id not in id_old2new_map_train.keys():
                if neibor_id in all_train_node_set:
                    id_old2new_map_train[neibor_id] = count_train
                    id_new2old_map_train[count_train] = neibor_id
                    # feat_data_train.append(feats_from_server[neibor_id])
                    count_train += 1

    # context.glContext.newToOldMap = id_new2old_map
    # context.glContext.oldToNewMap = id_old2new_map

    # 将顶点按照id_old2new_map转化
    nodes_train = [id_old2new_map_train[i] for i in train_node_old_id]

    idx_train_full=[id_old2new_map_train[i] for i in idx_train_old]

    # 将邻接表按照id_old2new_map转化
    # 将标签按照id_old2new_map转换
    adjs_train = []
    labels_train = []
    for new_id in range(len(train_node_old_id)):
        old_id = id_new2old_map_train[new_id]
        if old_id in all_train_node_set:
            neibor_set_old = adjs_from_server[old_id]
            neibor_set_new = []
            # 转化labels
            labels_train.append(labels_from_server[old_id])

            for item in neibor_set_old:
                if item in all_train_node_set:
                    neibor_set_new.append(id_old2new_map_train[item])
            adjs_train.append(neibor_set_new)

    labels_train = np.array(labels_train)

    feat_data_train = sp.csr_matrix(feat_data_train, dtype=np.float32)

    feat_data_train = torch.FloatTensor(np.array(feat_data_train.todense()))
    labels_train = torch.LongTensor(labels_train)
    graph_train = Graph(train_node_old_id, feat_data_train, labels_train,
                        adjs_train, id_old2new_map_train, id_new2old_map_train, first_hop_set_for_workers_train)

    idx_train=[id_old2new_map_train[i] for i in idx_train_old]
    idx_train = torch.LongTensor(idx_train)
    idx_train_full=torch.LongTensor(idx_train_full)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return {'idx_val': idx_val,
            'idx_train': idx_train,
            'idx_train_full':idx_train_full,
            'idx_test': idx_test,
            'idx_train_for_test':idx_train,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'train_num': train_num,
            'test_num': test_num,
            'val_num': val_num,
            'graph_full': graph_full,
            'graph_train': graph_train}

# Yu
def all_agg_node_num(nodes_from_server, idx_train, idx_val, idx_test, adj_lists_from_server):
    # 划分节点
    real_train_num = [nodes_from_server[i] for i in idx_train]
    real_val_num = [nodes_from_server[i] for i in idx_val]
    real_test_num = [nodes_from_server[i] for i in idx_test]

    worker_id = context.glContext.config['id']
    context.glContext.dgnnServerRouter[0].sendTrainNode(worker_id, real_train_num)  # 传输训练节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("send train node finish")

    context.glContext.dgnnServerRouter[0].sendValNode(worker_id, real_val_num)  # 传输验证节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("send val node finish")

    context.glContext.dgnnServerRouter[0].sendTestNode(worker_id, real_test_num)  # 传输测试节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("send test node finish")

    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("waiting other worker send node...")

    all_train_nodes = context.glContext.dgnnServerRouter[0].pullTrainNode()  # 获得所有的训练节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("get all train node!")
    all_val_nodes = context.glContext.dgnnServerRouter[0].pullValNode()  # 获得所有验证节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("get all val node!")
    all_test_nodes = context.glContext.dgnnServerRouter[0].pullTestNode()  # 获得所有测试节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("get all test node!")

    all_train_nodes_set = set(all_train_nodes)

    # 返回所有需要聚合的顶点
    return all_train_nodes_set


def get_needed_nodes(layid, adj, nodes,local_node_all):
    nodes_new = []
    for id in nodes:
        if local_node_all.__contains__(id):
            nodes_new.append(id)
            nodes_new.extend(adj[id])
            # for nei in adj[id]:
            #     nodes_new.add(nei)
    # nodes_new = list(nodes_new)
    context.glContext.dgnnServerRouter[0].sendNode(layid, nodes_new)
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    nodes_new_all = context.glContext.dgnnServerRouter[0].pullNode(layid)
    return set(nodes_new_all)


def all_agg_node_num_all(nodes_from_server, idx_train, adj_lists_from_server):
    # 划分节点
    real_train_num = [nodes_from_server[i] for i in idx_train]
    local_node_all=set(nodes_from_server)

    worker_id = context.glContext.config['id']
    context.glContext.dgnnServerRouter[0].sendTrainNode(worker_id, real_train_num)  # 传输训练节点
    print("send train node finish")

    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("waiting other worker send node...")

    all_train_nodes = context.glContext.dgnnServerRouter[0].pullTrainNode()  # 获得所有的训练节点
    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("get all train node!")

    all_train_nodes_set = set(all_train_nodes)
    # get needed vertices for each layer
    needed_nodes = []
    needed_nodes.append(all_train_nodes_set)

    for i in range(context.glContext.config['layerNum']):
        needed_nodes_layer = get_needed_nodes(i + 1, adj_lists_from_server, needed_nodes[i],local_node_all)
        needed_nodes.append(needed_nodes_layer)

    localNode = []
    needed_nodes_layer=needed_nodes[context.glContext.config['prune_layer']]
    # needed_nodes_layer=needed_nodes[1]
    nodes_set=set(nodes_from_server)
    for i in needed_nodes_layer:
        if nodes_set.__contains__(i):
            localNode.append(i)

    # 返回所有需要聚合的顶点
    return needed_nodes_layer, localNode
