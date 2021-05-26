import torch
import numpy as np
import scipy.sparse as sp
from cmake.build.example2 import *
from context import context
import random
import gc
import time
from data_structure.graph import Graph


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


def build_fsthop_list(v_set, adj_old_dict, node2worker_old_dict):
    fsthop_set = set()
    for id in v_set:
        for nid in adj_old_dict[id]:
            fsthop_set.add(int(nid))

    fsthop_for_worker_list = []
    for i in range(context.glContext.config['worker_num']):
        fsthop_for_worker_list.append(list())
    for id in fsthop_set:
        wid = node2worker_old_dict[id]
        fsthop_for_worker_list[wid].append(id)

    # transform set to numpy ndarray
    for i in range(len(fsthop_for_worker_list)):
        fsthop_for_worker_list[i] = np.array(fsthop_for_worker_list[i])
    return fsthop_for_worker_list


def build_feat_and_iddict(feat_old_dict, agg_node_old, fsthop_old_for_worker_list):
    feat_list = []
    id_old2new_dict = {}
    id_new2old_dict = {}
    count = 0
    for id in agg_node_old:
        feat_list.append(feat_old_dict[id])
        id_old2new_dict[id] = count
        id_new2old_dict[count] = id
        count += 1

    for wid in range(len(fsthop_old_for_worker_list)):
        if wid==context.glContext.config['id']:
            for neibor_id in fsthop_old_for_worker_list[wid]:
                if neibor_id not in id_old2new_dict.keys():
                    feat_list.append(list)
                    id_old2new_dict[neibor_id] = count
                    id_new2old_dict[count] = neibor_id
                    count += 1


    for wid in range(len(fsthop_old_for_worker_list)):
        if wid!=context.glContext.config['id']:
            for neibor_id in fsthop_old_for_worker_list[wid]:
                if neibor_id not in id_old2new_dict.keys():
                    id_old2new_dict[neibor_id] = count
                    id_new2old_dict[count] = neibor_id
                    count += 1

    for neibor_id in fsthop_old_for_worker_list[context.glContext.config['id']]:
        newid=id_old2new_dict[neibor_id]
        feat_list[newid]=feat_old_dict[neibor_id]



    context.glContext.newToOldMap = id_new2old_dict
    context.glContext.oldToNewMap = id_old2new_dict
    feat_tensor = torch.FloatTensor(np.array(feat_list))
    return feat_tensor, id_old2new_dict, id_new2old_dict


def build_adj_and_label(v_num, id_new2old_dict, adj_old_dict, label_old_list, id_old2new_dict):
    adjs = []
    labels = []
    for new_id in range(v_num):
        old_id = id_new2old_dict[new_id]
        neibor_set_old = adj_old_dict[old_id]
        neibor_set_new = []
        # 转化labels
        labels.append(label_old_list[old_id])

        for item in neibor_set_old:
            if item in id_old2new_dict.keys():
                neibor_set_new.append(id_old2new_dict[item])
        adjs.append(neibor_set_new)
    labels = torch.LongTensor(labels)
    return adjs, labels


# Yu
def load_datav2(dgnnClient):
    global item
    isRandomData = True
    np.random.seed(1)
    random.seed(1)

    # 这里读取的都是按照server统一编号的；
    v_old_list = dgnnClient.nodes
    feat_old_dict = dgnnClient.features
    label_old_list = dgnnClient.labels
    adj_old_dict = dgnnClient.adjs
    worker_contain_node_old_list = dgnnClient.nodesForEachWorker
    node2worker_old_dict = {}

    dgnnClient.freeSpace()

    # build node2worker_dict
    for i in range(context.glContext.config['worker_num']):
        for j in range(len(worker_contain_node_old_list[i])):
            node2worker_old_dict[worker_contain_node_old_list[i][j]] = i

    dgnnClient.layerNum = context.glContext.config['layerNum']

    # 开始定义图神经网络模型
    v_local_num = len(v_old_list)

    train_ratio = context.glContext.config['train_num'] / context.glContext.config['data_num']
    val_ratio = context.glContext.config['val_num'] / context.glContext.config['data_num']
    test_ratio = context.glContext.config['test_num'] / context.glContext.config['data_num']

    train_num = int(v_local_num * train_ratio)
    val_num = int(v_local_num * val_ratio)
    test_num = int(v_local_num * test_ratio)

    rand_indices = None
    if isRandomData:
        rand_indices = np.random.permutation(v_local_num)  # 随机索引
    else:
        rand_indices = np.arange(v_local_num)

    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    feat_dim = context.glContext.config['feature_dim']
    all_agg_node_old,  all_agg_node_old_train, all_agg_node_old_val, all_agg_node_old_test = all_agg_node_num(v_old_list,
                                                                                                             idx_train,
                                                                                                             idx_val,
                                                                                                             idx_test,
                                                                                                             adj_old_dict)
    all_agg_node_old = sorted(all_agg_node_old)
    all_agg_node_old_train = sorted(all_agg_node_old_train)
    all_agg_node_old_val = sorted(all_agg_node_old_val)
    all_agg_node_old_test = sorted(all_agg_node_old_test)

    # 先构建整体的adj, label, feat, first_hop_neighbor_set
    fsthop_old_for_worker_list = build_fsthop_list(all_agg_node_old, adj_old_dict, node2worker_old_dict)
    feat_new_tensor, id_old2new_dict, id_new2old_dict = build_feat_and_iddict(feat_old_dict, all_agg_node_old,
                                                                              fsthop_old_for_worker_list)
    v_new_list = [id_old2new_dict[i] for i in v_old_list]
    adj_new_list, label_new_tensor = build_adj_and_label(v_local_num, id_new2old_dict, adj_old_dict, label_old_list,
                                                         id_old2new_dict)
    full_graph = Graph(all_agg_node_old, feat_new_tensor, label_new_tensor, adj_new_list, id_old2new_dict,
                       id_new2old_dict,
                       fsthop_old_for_worker_list)

    #  build training graph, where old_id is the initial id from master,
    #  and new_id is built according to different stages

    fsthop_old_for_worker_list_train = build_fsthop_list(all_agg_node_old_train, adj_old_dict, node2worker_old_dict)
    feat_new_tensor_train, id_old2new_dict_train, id_new2old_dict_train = build_feat_and_iddict(feat_old_dict,
                                                                                                all_agg_node_old_train,
                                                                                                fsthop_old_for_worker_list_train)
    all_agg_node_old_train.clear()
    all_agg_node_old_train=[id_new2old_dict_train[i] for i in range(len(id_new2old_dict_train))]
    v_new_list_train = [id_old2new_dict_train[i] for i in all_agg_node_old_train]
    adj_new_list_train, label_new_tensor_train = build_adj_and_label(len(all_agg_node_old_train), id_new2old_dict_train, adj_old_dict, label_old_list,
                                                         id_old2new_dict_train)


    train_graph=Graph(all_agg_node_old_train, feat_new_tensor_train, label_new_tensor_train, adj_new_list_train, id_old2new_dict_train,
                      id_new2old_dict_train,
                      fsthop_old_for_worker_list_train)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return {'idx_val': idx_val,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'train_num': train_num,
            'test_num': test_num,
            'val_num': val_num,
            'full_graph':full_graph,
            'train_graph': train_graph}


# Yu
def all_agg_node_num(v_old_list, idx_train, idx_val, idx_test, adj_old_dict):
    # 划分节点
    real_train_num = [v_old_list[i] for i in idx_train]
    real_val_num = [v_old_list[i] for i in idx_val]
    real_test_num = [v_old_list[i] for i in idx_test]

    # 本地的一阶邻居也属于train
    neib_set=set()
    for id in real_train_num:
        for nid in adj_old_dict[id]:
            neib_set.add(nid)
    for id in neib_set:
        if real_train_num.count(id)==0:
            real_train_num.append(id)

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
    # local nodes that other workers use in training stage, need to be added in remote_hop_train_set
    for neiborSet in adj_old_dict:
        for neiborId in adj_old_dict[neiborSet]:
            if neiborId in set_remote_train_node:
                remote_hop_train_set.add(neiborSet)
            if neiborId in set_remote_val_node:
                remote_hop_val_set.add(neiborSet)
            if neiborId in set_remote_test_node:
                remote_hop_test_set.add(neiborSet)
    # 返回所有需要聚合的顶点
    return v_old_list, remote_hop_train_set | set(real_train_num), set(real_val_num) | remote_hop_val_set, set(
        real_test_num) | remote_hop_test_set
