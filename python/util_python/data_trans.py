import torch
import numpy as np
import scipy.sparse as sp
from cmake.build.example2 import *
from context import context
import random
import gc
import time


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


def load_data(dgnnClient):
    isRandomData = True
    np.random.seed(1)
    random.seed(1)

    # 这里读取的都是按照master统一编号的,
    nodes_from_server = dgnnClient.nodes
    feats_from_server = dgnnClient.features
    labels_from_server = dgnnClient.labels
    adjs_from_server = dgnnClient.adjs

    worker_contains_nodes = dgnnClient.nodesForEachWorker
    nodes_in_worker = {}

    dgnnClient.freeSpace()

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

    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    # build the first-hop neighboring set (containing the local and remote neighbors)
    first_hop_set = set()
    for neighbor_set in adjs_from_server:
        for neighbor_id in adjs_from_server[neighbor_set]:
            first_hop_set.add(int(neighbor_id))

    first_hop_set_for_workers = context.glContext.config['firstHopForWorkers']
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
    del feats_from_server
    gc.collect()
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

    del adjs_from_server
    gc.collect()
    labels = np.array(labels)

    # feat_data,nodes,adjs,labels重置了顺序
    # 其中feat_data包含了临界的一阶邻居的特征，而nodes里的id就对应了feat_data, adjs, labels对应维度

    # features = nn.Embedding(len(feat_data), feat_dim)  # 构建特征嵌入矩阵
    # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)  # 构建特征值权重矩阵
    features = sp.csr_matrix(feat_data, dtype=np.float32)
    # features=normalize_feature(features)

    # features = normalize(features)

    # 将邻接矩阵处理成coo的格式
    # edges=[[0,5],[0,10],[0,100],[2,100],[5,1002]]
    # edges=np.array(edges)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return {'features': features,
            'adjs': adjs,
            "nodes": nodes,
            'labels': labels,
            'nodes_from_server': nodes_from_server,
            'firstHopSetsForWorkers': first_hop_set_for_workers,
            'idx_val': idx_val,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'train_num': train_num,
            'test_num': test_num,
            'val_num': val_num,
            'id_old2new_map': id_old2new_map,
            'id_new2old_map': id_new2old_map}
# Yu
def load_datav2(dgnnClient):
    global item
    isRandomData = True
    np.random.seed(1)
    random.seed(1)

    # 这里读取的都是按照server统一编号的；
    nodes_from_server = dgnnClient.nodes
    feat_data_dict_from_server = dgnnClient.features
    labels_from_server = dgnnClient.labels
    adj_lists_from_server = dgnnClient.adjs
    nodesForEachWorker = dgnnClient.nodesForEachWorker
    nodeToWorkerMap = {}

    dgnnClient.freeSpace()

    # build nodeToWorker map
    for i in range(context.glContext.config['worker_num']):
        for j in range(len(nodesForEachWorker[i])):
            nodeToWorkerMap[nodesForEachWorker[i][j]] = i

    dgnnClient.layerNum = context.glContext.config['layerNum']

    # 开始定义图神经网络模型
    data_num = len(nodes_from_server)

    train_ratio = context.glContext.config['train_num'] / context.glContext.config['data_num']
    val_ratio = context.glContext.config['val_num'] / context.glContext.config['data_num']
    test_ratio = context.glContext.config['test_num'] / context.glContext.config['data_num']

    train_num = int(data_num * train_ratio)
    val_num = int(data_num * val_ratio)
    test_num = int(data_num * test_ratio)

    rand_indices = None
    if isRandomData:
        rand_indices = np.random.permutation(data_num)  # 随机索引
    else:
        rand_indices = np.arange(data_num)

    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    feat_dim = context.glContext.config['feature_dim']
    all_agg_node , all_agg_node_train, all_agg_node_val, all_agg_node_test = all_agg_node_num(nodes_from_server,idx_train,idx_val,idx_test,adj_lists_from_server)

    # 构建本地顶点的一阶邻居集合 change
    # firstHopSet = set()
    # for neiborSet in adj_lists_from_server:
    #     for neiborId in adj_lists_from_server[neiborSet]:
    #         firstHopSet.add(int(neiborId))

    isTrain = True
    if(isTrain):
        firstHopSet = set()
        for neiborSet in all_agg_node:
            for neiborId in adj_lists_from_server[neiborSet]:
                firstHopSet.add(int(neiborId))

    firstHopSetsForWorkers = context.glContext.config['firstHopForWorkers']
    for i in range(context.glContext.config['worker_num']):
        firstHopSetsForWorkers.append(list())
    for id in firstHopSet:
        # workerId = id % context.glContext.config['worker_num']
        # firstHopSetsForWorkers[workerId].append(id)
        workerId = nodeToWorkerMap[id]
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
    del feat_data_dict_from_server
    gc.collect()


    #对一阶邻居编码
    # for vid in adj_lists_from_server:
    #     for neibor_id in adj_lists_from_server[vid]:
    #         if neibor_id not in id_old2new_map.keys():
    #             id_old2new_map[neibor_id] = count
    #             id_new2old_map[count] = neibor_id
    #             count += 1
    # change
    for vid in all_agg_node:
        for neibor_id in adj_lists_from_server[vid]:
            if neibor_id not in id_old2new_map.keys():
                id_old2new_map[neibor_id] = count
                id_new2old_map[count] = neibor_id
                count += 1

    context.glContext.newToOldMap = id_new2old_map
    context.glContext.oldToNewMap = id_old2new_map

    # neb_train_node, neb_val_node, neb_test_node = remote_neb_node(nodes_from_server, idx_train, idx_val, idx_test,
    #                                                               adj_lists_from_server, id_old2new_map)
    # print("neb_train_node:", len(neb_train_node))
    # print("neb_val_node:", len(neb_val_node))
    # print("neb_test_node", len(neb_test_node))

    # 将顶点按照id_old2new_map转化
    nodes = [id_old2new_map[i] for i in nodes_from_server]
    #d_sort_1=sorted(d,key=lambda x:(x[0],x[1]))#都是从小到大排列的
    #有必要吗？
    all_agg_node  = sorted(all_agg_node)
    all_agg_node_train = sorted(all_agg_node_train)
    all_agg_node_val = sorted(all_agg_node_val)
    all_agg_node_test = sorted(all_agg_node_test)

    agg_node = {}
    agg_node[0] = [id_old2new_map[i] for i in all_agg_node]
    agg_node[1] = [id_old2new_map[i] for i in all_agg_node_train]
    agg_node[2] = [id_old2new_map[i] for i in all_agg_node_val]
    agg_node[3] = [id_old2new_map[i] for i in all_agg_node_test]
    # 将邻接表按照id_old2new_map转化
    # 将标签按照id_old2new_map转换
    adjs = []
    labels = []
    # for new_id in range(data_num):
    #     old_id = id_new2old_map[new_id]
    #     neibor_set_old = adj_lists_from_server[old_id]
    #     neibor_set_new = []
    #     # 转化labels
    #     labels.append(labels_from_server[old_id])
    #
    #     for item in neibor_set_old:
    #         neibor_set_new.append(id_old2new_map[item])
    #     adjs.append(neibor_set_new)

    #change
    for new_id in range(data_num):
        old_id = id_new2old_map[new_id]
        neibor_set_old = adj_lists_from_server[old_id]
        neibor_set_new = []
        # 转化labels
        labels.append(labels_from_server[old_id])

        for item in neibor_set_old:
            if item in id_old2new_map.keys():
                neibor_set_new.append(id_old2new_map[item])
        adjs.append(neibor_set_new)
    #change
    # train_adjs = []
    # train_labels = []
    # for old_id in all_agg_node:
    #     neibor_set_old = adj_lists_from_server[old_id]
    #     neibor_set_new = []
    #     # 转化labels
    #     train_labels.append(labels_from_server[old_id])
    #
    #     for item in neibor_set_old:
    #         neibor_set_new.append(id_old2new_map[item])
    #     train_adjs.append(neibor_set_new)
    #
    # del adj_lists_from_server
    # gc.collect()
    # labels = np.array(labels)

    # feat_data,nodes,adjs,labels重置了顺序
    # 其中feat_data包含了临界的一阶邻居的特征，而nodes里的id就对应了feat_data, adjs, labels对应维度

    # features = nn.Embedding(len(feat_data), feat_dim)  # 构建特征嵌入矩阵
    # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)  # 构建特征值权重矩阵
    features = sp.csr_matrix(feat_data, dtype=np.float32)
    # features=normalize_feature(features)

    # features = normalize(features)

    # 将邻接矩阵处理成coo的格式
    # edges=[[0,5],[0,10],[0,100],[2,100],[5,1002]]
    # edges=np.array(edges)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return {'features': features,
            'adjs': adjs,
            "nodes": nodes,
            'labels': labels,
            'nodes_from_server': nodes_from_server,
            'firstHopSetsForWorkers': firstHopSetsForWorkers,
            'idx_val': idx_val,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'train_num': train_num,
            'test_num': test_num,
            'val_num': val_num,
            'id_old2new_map': id_old2new_map,
            'id_new2old_map': id_new2old_map,
            'agg_node': agg_node}

# Yu
def all_agg_node_num(nodes_from_server, idx_train, idx_val, idx_test, adj_lists_from_server):
    # 划分节点
    real_train_num = [nodes_from_server[i] for i in idx_train]
    real_val_num = [nodes_from_server[i] for i in idx_val]
    real_test_num = [nodes_from_server[i] for i in idx_test]

    worker_id = context.glContext.config['id']
    context.glContext.dgnnServerRouter[0].sendTrainNode(worker_id, real_train_num)  # 传输训练节点
    print("send train node finish")
    context.glContext.dgnnServerRouter[0].sendValNode(worker_id, real_val_num)  # 传输验证节点
    print("send val node finish")
    context.glContext.dgnnServerRouter[0].sendTestNode(worker_id, real_test_num)  # 传输测试节点
    print("send test node finish")

    context.glContext.dgnnServerRouter[0].server_Barrier(0)  # 等待所有节点都上传完训练节点
    print("waiting other worker send node...")

    all_train_nodes = context.glContext.dgnnServerRouter[0].pullTrainNode()  # 获得所有的训练节点
    print("get all train node!")
    all_val_nodes = context.glContext.dgnnServerRouter[0].pullValNode()  # 获得所有验证节点
    print("get all val node!")
    all_test_nodes = context.glContext.dgnnServerRouter[0].pullTestNode()  # 获得所有测试节点
    print("get all test node!")

    # 选出一阶邻居为其他机器上是训练节点的点
    set_local_train_node = set(real_train_num)
    set_remote_train_node = set(all_train_nodes) - set_local_train_node

    set_local_val_node = set(real_val_num)
    set_remote_val_node = set(all_val_nodes) - set_local_val_node

    set_local_test_node = set(real_test_num)
    set_remote_test_node = set(all_test_nodes) - set_local_test_node

    remote_hop_train_set = set()
    remote_hop_val_set = set()
    remote_hop_test_set = set()
    for neiborSet in adj_lists_from_server:
        for neiborId in adj_lists_from_server[neiborSet]:
            if neiborId in set_remote_train_node:
                remote_hop_train_set.add(neiborSet)
            if neiborId in set_remote_val_node:
                remote_hop_val_set.add(neiborSet)
            if neiborId in set_remote_test_node:
                remote_hop_test_set.add(neiborSet)
    # 返回所有需要聚合的顶点
    return remote_hop_train_set | set(real_train_num) | remote_hop_val_set | remote_hop_test_set, remote_hop_train_set | set(real_train_num),  set(real_val_num) | remote_hop_val_set, set(real_test_num) | remote_hop_test_set
