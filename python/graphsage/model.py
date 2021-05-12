import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import argparse
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from dist_sage.encoders import Encoder
from dist_sage.aggregators import MeanAggregator
import context.context as context
import util_python.param_parser as pp
import context.store as store
from cmake.build.example2 import *
import gc
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import util_python.data_trans as dt

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):  # 有监督Graphsage

    def __init__(self, num_classes, enc, layer_id):  # num_classes为分类数 enc为嵌入
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()  # 损失函数设为交叉熵损失
        self.layer_id=layer_id
        self.weight=[]
        # self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))  # 构建权重矩阵
        # nn.init.xavier_uniform_(self.weight)  # 以均匀分布初始化

    def forward(self, nodes):
        embeds = self.enc(nodes)  # 节点嵌入
        for i in range(context.glContext.config['server_num']):
            self.weight.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(self.layer_id))
        self.weight=torch.FloatTensor(self.weight)
        self.weight.requires_grad=True
        self.weight.retain_grad()
        scores = self.weight.t().mm(embeds)  # 分数（节点的表示向量)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())  # 返回交叉熵损失


def printInfo(firstHopSetsForWorkers):
    print("local and remote vertex distribution:")
    print("local worker {0}:{1}".format(context.glContext.config['id'],
                                        len(firstHopSetsForWorkers[context.glContext.config['id']])))
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.config['id']:
            print("worker {0}:{1}".format(i, len(firstHopSetsForWorkers[i])))

def run_gnn(dgnnClient):
    # data processing
    data=dt.load_data(dgnnClient)
    feat_data=data['features']
    adjs=data['adjs']
    nodes_from_server=data['nodes_from_server']
    firstHopSetsForWorkers=data['firstHopSetsForWorkers']
    labels=data['labels']
    idx_val=data['idx_val']
    idx_train=data['idx_train']
    idx_test=data['idx_test']
    id_old2new_map=data['id_old2new_map']
    id_new2old_map=data['id_new2old_map']
    nodes=data['nodes']
    train_ratio=data['train_ratio']
    test_ratio=data['test_ratio']
    val_ratio=data['val_ratio']
    train_num=data['train_num']
    test_num=data['test_num']
    val_num=data['val_num']
    ifCompress = context.glContext.config['ifCompress']
    nodes=list(nodes)
    idx_val=list(idx_val.detach().numpy())
    idx_train=list(idx_train.detach().numpy())
    idx_test=list(idx_test.detach().numpy())

    printInfo(firstHopSetsForWorkers)

    features = nn.Embedding(len(nodes), context.glContext.config['feature_dim'])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)  # 第一层聚合
    enc1 = Encoder(features, context.glContext.config['feature_dim'], context.glContext.config['hidden'][0], adjs, agg1, num_sample=5,layer_id=0,gcn=True, cuda=False)  # 第一层嵌入
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)  # 第二层聚合,.t() transposition
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, context.glContext.config['hidden'][1], adjs, agg2,num_sample=5,layer_id=1,
                   base_model=enc1, gcn=True, cuda=False)  # 第二层嵌入
    # enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 64, adj_lists, agg2,
    #                gcn=True, cuda=False)  # 第二层嵌入
    # agg3 = MeanAggregator(lambda nodes: enc2(nodes).t(), cuda=False)
    # enc3 = Encoder(lambda nodes: enc2(nodes).t(), enc2.embed_dim, 128, adj_lists, agg3, 10,
    #                base_model=enc1, gcn=True, cuda=False)  # third layer embeddings

    # dist_sage = SupervisedGraphSage(7, enc3)
    graphsage = SupervisedGraphSage(context.glContext.config['class_num'], enc2,layer_id=2)
    #    dist_sage.cuda()


    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dist_sage.parameters()), lr=0.1)  # 使用SGD进行优化
    print(list(graphsage.parameters()))
    times = []
    for epoch in range(context.glContext.config['IterNum']):
        start_time = time.time()
        # optimizer.zero_grad()  # 清空过往梯度
        loss = graphsage.loss(idx_train,
                              Variable(torch.LongTensor(labels[np.array(idx_train)])))
        loss.backward()  # 反向传播，计算当前梯度
        print(graphsage.weight.grad)
        # print("weight:")
        # print(dist_sage.weight)
        # print("grad:")
        # print(dist_sage.weight.grad)
        # print(enc2.weight.grad)
        # print(enc1.weight.grad)
        # dist_sage.weight.grad[0, 0] = 1000000
        # optimizer.step()  # 根据梯度更新网络参数
        end_time = time.time()
        times.append(end_time - start_time)
        print(epoch, loss.item())  # 输出轮次、loss值
        val_output = graphsage.forward(idx_val)  # 所有顶点的表示向量
        print("Validation F1:", f1_score(labels[idx_val], val_output.data.numpy().argmax(axis=1), average="micro"))  # 输出F1值
        test_output = graphsage.forward(idx_test)  # 所有顶点的表示向量
        print("test F1:", f1_score(labels[idx_test], test_output.data.numpy().argmax(axis=1), average="micro"))  # 输出F1值

    # print("Average batch time:", np.mean(times))  # 每轮平均时间

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
    pp.parserInit()
    pp.printContext()

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
        # context.glContext.dgnnMasterRouter.pullDataFromMaster(
        #     id, context.glContext.config['worker_num'],
        #     context.glContext.config['data_num'],
        #     context.glContext.config['data_path'],
        #     context.glContext.config['feature_dim'],
        #     context.glContext.config['class_num'])

        context.glContext.dgnnMasterRouter.pullDataFromMasterGeneral(
            id, context.glContext.config['worker_num'],
            context.glContext.config['data_num'],
            context.glContext.config['data_path'],
            context.glContext.config['feature_dim'],
            context.glContext.config['class_num'],
            context.glContext.config['partitionMethod'],
            context.glContext.config['edge_num'])

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

        context.glContext.dgnnClient.initCompressBitMap(context.glContext.config['bitNum'])
        context.glContext.dgnnServerRouter[0].server_Barrier(0)
        if id == 0:
            context.glContext.dgnnMasterRouter.freeMaster()

        # # 已经将各自数据放到了数据库里，接下来定义GNN模型，然后训练
        # # 这里的操作的workerStore是全局静态变量，因此整个进程都可见
        run_gnn(context.glContext.dgnnClient)

    elif context.glContext.config['role'] == 'master':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)
        # ServiceImpl.RunServerByPy('127.0.0.1:4001')