import sys, os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH+'/../')
print(BASE_PATH)

import torch
import torch.nn as nn

import numpy as np
import time

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


from context import context as context
from util_python import param_parser as pp
from context import store as store
from cmake.build.example2 import *

import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp

import autograd.autograd as atg
import util_python.metric as metric
from dist_sage.models import GraphSAGE

import util_python.remote_access as ra
import util_python.data_trans as dt
import util_python.param_util as pu
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
        self.weight=[]
        for i in range(context.glContext.config['server_num']):
            self.weight.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(self.layer_id))
        self.weight=torch.FloatTensor(self.weight)
        self.weight.requires_grad=True
        self.weight.retain_grad()
        atg.Z3=embeds

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

def run_gnn(dgnnClient,model):
    # data processing
    data=dt.load_data(dgnnClient)
    features=data['features']
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

    edges = []
    # 从adj中解析出edge
    for i in range(len(adjs)):
        for nei_id in adjs[i]:
            edges.append([i, nei_id])

    edges = np.array(edges)
    adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(len(nodes), len(id_old2new_map)),
                         dtype=np.int)
    adjs=dt.normalize(adjs)

    adjs = dt.sparse_mx_to_torch_sparse_tensor(adjs)  # 邻接矩阵转为tensor处理

    printInfo(firstHopSetsForWorkers)

    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=8e-4)

    ifCompress = context.glContext.config['ifCompress']
    timeList=[]
    for epoch in range(context.glContext.config['IterNum']):
        startTimeTotle = time.time()
        model.train()
        optimizer.zero_grad()
        store.isTrain = True
        if ifCompress:
            context.glContext.config['ifCompress'] = True

        # slow
        start = time.time()
        output = model(features, adjs, nodes_from_server, epoch)
        end = time.time()
        # print("output time:{0}".format(end - start))

        start_othertime = time.time()
        if context.glContext.config['isNeededExactBackProp']:
            if context.glContext.config["layerNum"] == 2:
                atg.H2 = output
                atg.Z2.required_grad = True
                atg.Z2.retain_grad()
            elif context.glContext.config["layerNum"] == 3:
                atg.H3 = output
                atg.Z3.required_grad = True
                atg.Z3.retain_grad()
            elif context.glContext.config["layerNum"] == 4:
                atg.H4 = output
                atg.Z4.required_grad = True
                atg.Z4.retain_grad()

        # print(output)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，
        # 这里就要使用CrossEntropyLoss了
        # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
        # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
        # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
        # https://blog.csdn.net/hao5335156/article/details/80607732
        acc_train = metric.accuracy(output[idx_train], labels[idx_train])  # 计算准确率
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
                ra.pullNeighborG(nodes, epoch, 2)
                end = time.time()
                # if context.glContext.ifShowInfo:
                # print("pull neiborgh G time:{0}".format(end - start))

                a = torch.spmm(adjs, atg.G2)
                b = torch.mm(a, model.gc2.weight.t())
                atg.G1 = torch.mul(b, atg.sigma_z1_grad)
                atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)

                np.set_printoptions(threshold=sys.maxsize)

                model.gc1.weight.grad.data = atg.Y0

                model.gc2.weight.grad.data = atg.Y1




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


                atg.sigma_z2_grad = atg.Z2.data
                atg.sigma_z2_grad = torch.tensor(np.where(atg.sigma_z2_grad > 0, 1, 0))
                # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
                # 这段跟获取一阶邻居的嵌入表示类似
                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                ra.pullNeighborG(nodes, epoch, 3)
                a = torch.spmm(adjs, atg.G3)
                b = torch.mm(a, model.gc3.weight.t())
                atg.G2 = torch.mul(b, atg.sigma_z2_grad)
                atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)


                # set G2
                G_list = atg.G2.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]
                dgnnClient.setG(G_dict, 2)

                atg.sigma_z1_grad = atg.Z1.data
                atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))

                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                ra.pullNeighborG(nodes, epoch, 2)
                a = torch.spmm(adjs, atg.G2)
                b = torch.mm(a, model.gc2.weight.t())
                atg.G1 = torch.mul(b, atg.sigma_z1_grad)
                atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)


                np.set_printoptions(threshold=sys.maxsize)
                model.gc1.weight.grad.data = atg.Y0

                model.gc2.weight.grad.data = atg.Y1

                model.gc3.weight.grad.data = atg.Y2


            elif context.glContext.config["layerNum"] == 4:
                atg.G4 = atg.Z4.grad
                # 算出G2后，给自己所在的server赋值
                G_list = atg.G4.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]

                dgnnClient.setG(G_dict, 4)

                # G4本地完全的，不需要从其他机器获取
                atg.Y3 = torch.mm(atg.A_X_H3.t(), atg.G4)


                atg.sigma_z3_grad = atg.Z3.data
                atg.sigma_z3_grad = torch.tensor(np.where(atg.sigma_z3_grad > 0, 1, 0))
                # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
                # 这段跟获取一阶邻居的嵌入表示类似
                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                ra.pullNeighborG(nodes, epoch, 4)
                a = torch.spmm(adjs, atg.G4)
                b = torch.mm(a, model.gc4.weight.t())
                atg.G3 = torch.mul(b, atg.sigma_z3_grad)
                atg.Y2 = torch.mm(atg.A_X_H2.t(), atg.G3)


                # set G3
                G_list = atg.G3.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]
                dgnnClient.setG(G_dict, 3)

                atg.sigma_z2_grad = atg.Z2.data
                atg.sigma_z2_grad = torch.tensor(np.where(atg.sigma_z2_grad > 0, 1, 0))

                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                ra.pullNeighborG(nodes, epoch, 3)
                a = torch.spmm(adjs, atg.G3)
                b = torch.mm(a, model.gc3.weight.t())
                atg.G2 = torch.mul(b, atg.sigma_z2_grad)
                atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)


                # set G2
                G_list = atg.G2.detach().numpy().tolist()
                G_dict = {}
                for i in range(len(G_list)):
                    G_dict[id_new2old_map[i]] = G_list[i]
                dgnnClient.setG(G_dict, 2)

                atg.sigma_z1_grad = atg.Z1.data
                atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))

                context.glContext.dgnnServerRouter[0].server_Barrier(0)
                ra.pullNeighborG(nodes, epoch, 2)
                a = torch.spmm(adjs, atg.G2)
                b = torch.mm(a, model.gc2.weight.t())
                atg.G1 = torch.mul(b, atg.sigma_z1_grad)
                atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)


                np.set_printoptions(threshold=sys.maxsize)
                model.gc1.weight.grad.data = atg.Y0

                model.gc2.weight.grad.data = atg.Y1

                model.gc3.weight.grad.data = atg.Y2

                model.gc4.weight.grad.data = atg.Y3


        # 求出梯度后，发送到参数服务器中进行聚合，并更新参数值
        # a=model.gc1.weight.grad
        # 将权重梯度和偏移梯度分别转化成map<int,vector<vector<float>>>和map<int,vector<float>>
        weights_grad_map = {}
        bias_grad_map = {}

        # 这里先不做通用的，直接假定是两层，之后再改
        if context.glContext.config["layerNum"] == 2:
            weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
            weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()

        elif context.glContext.config["layerNum"] == 3:
            weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
            weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
            weights_grad_map[2] = model.gc3.weight.grad.detach().numpy().tolist()

        elif context.glContext.config["layerNum"] == 4:
            weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
            weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
            weights_grad_map[2] = model.gc3.weight.grad.detach().numpy().tolist()
            weights_grad_map[3] = model.gc4.weight.grad.detach().numpy().tolist()

        # print(weights_grad_map[0])
        # print(weights_grad_map[1])
        # print(bias_grad_map[0])
        # print(bias_grad_map[1])

        end_othertime = time.time()
        # print("other time:{0}".format(end_othertime - start_othertime))

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
                                                                          bias_grad_map, context.glContext.config['lr'])
            else:
                context.glContext.dgnnServerRouter[i].sendAndUpdateModels(context.glContext.config['id'], i,
                                                                          weights_grad_map_each,
                                                                          bias_grad_map, context.glContext.config['lr'])

        # 返回参数后同步,这里barrier的参数暂时没有用到
        context.glContext.dgnnServerRouter[0].server_Barrier(0)

        end = time.time()
        # print("update model time:{0}".format(end - start))
        endTimeTotle = time.time()
        timeList.append(endTimeTotle-startTimeTotle)

        # optimizer.step()  # 更新所有的参数  Gradient Descent
        model.eval()
        # output = model(features, adjs,nodes_from_server)

        if epoch % 10 == 0:
            context.glContext.config['ifCompress'] = False
            store.isTrain = False
            output = model(features, adjs, nodes_from_server, 10000)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 验证集的损失函数
            # acc_val = metric.accuracy(output[idx_val], labels[idx_val])
            acc_val=accuracy_score(labels[idx_val].detach().numpy(), output[idx_val].detach().numpy().argmax(axis=1))
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy_score(labels[idx_train].detach().numpy(), output[idx_train].detach().numpy().argmax(axis=1))
            # acc_train = metric.accuracy(output[idx_train], labels[idx_train])  # 计算准确率
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])  # 验证集的损失函数
            acc_test = metric.accuracy(output[idx_test], labels[idx_test])


            val_f1 = f1_score(labels[idx_val].detach().numpy(), output[idx_val].detach().numpy().argmax(axis=1),
                              average='micro')

            test_f1 = f1_score(labels[idx_test].detach().numpy(), output[idx_test].detach().numpy().argmax(axis=1),
                               average='micro')

            train_datanum_entire = int(train_ratio * context.glContext.config['data_num'])
            val_datanum_entire = int(val_ratio * context.glContext.config['data_num'])
            test_datanum_entire = int(test_ratio * context.glContext.config['data_num'])
            acc_entire = context.glContext.dgnnServerRouter[0].sendAccuracy(acc_val * val_num, acc_train * train_num,
                                                                            acc_test * test_num, val_f1 * val_num,
                                                                            test_f1 * test_num)
            # print('Epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'acc_train: {:.4f}'.format(acc_train.item()),
            #       'acc_val: {:.4f}'.format(acc_val.item()),
            #       'acc_train: {:.4f}'.format(acc_test.item()),
            #       "iter_time: {0}".format(endTimeTotle - startTimeTotle))

            print('Epoch: {:04d}'.format(epoch + 1),
                'acc_train_entire{:.4f}'.format(acc_entire['train'] / (float(train_datanum_entire))),
                  'acc_val_entire:{:.4f}'.format(acc_entire['val'] / (float(val_datanum_entire))),
                  'acc_test_entire: {:.4f}'.format(acc_entire['test'] / (float(test_datanum_entire))),
                  'f1_val_entire:{:.4f}'.format(acc_entire['val_f1'] / (float(val_datanum_entire))),
                  'f1_test_entire: {:.4f}'.format(acc_entire['test_f1'] / (float(test_datanum_entire))),
                  "epoch_iter_time: {0}".format(endTimeTotle - startTimeTotle),
                  "avg_iter_time: {0}".format(np.array(timeList).sum(axis=0)/len(timeList)))



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
        model = GraphSAGE(nfeat=context.glContext.config['feature_dim'],
                          nhid=context.glContext.config['hidden'],
                          nclass=context.glContext.config['class_num'],
                          dropout=0.5)

        if context.glContext.config['id'] == 0:
            pu.assignParam()
            for i in range(context.glContext.config['server_num']):
                context.glContext.dgnnServerRouter[i].initParameter(
                    context.glContext.config['worker_num'],
                    context.glContext.config['server_num'],
                    context.glContext.config['feature_dim'],
                    context.glContext.config['hidden'],
                    context.glContext.config['class_num'],
                    id,
                    context.glContext.weightForServer[i],
                    context.glContext.bias
                )

        context.glContext.dgnnClient.initCompressBitMap(context.glContext.config['bitNum'])
        context.glContext.dgnnServerRouter[0].server_Barrier(0)
        if id == 0:
            context.glContext.dgnnMasterRouter.freeMaster()

        # # 已经将各自数据放到了数据库里，接下来定义GNN模型，然后训练
        # # 这里的操作的workerStore是全局静态变量，因此整个进程都可见
        run_gnn(context.glContext.dgnnClient,model)

    elif context.glContext.config['role'] == 'master':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)
        # ServiceImpl.RunServerByPy('127.0.0.1:4001')