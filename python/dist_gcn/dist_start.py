import sys, os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
print(BASE_PATH)
import time
import numpy as np
from sklearn.metrics import f1_score
import util_python.param_parser as pp
import util_python.metric as metric
from sklearn.metrics import accuracy_score
# BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_PATH)
# # sys.path.insert(1, BASE_PATH + '/../')

import torch.nn.functional as F
import scipy.sparse as sp
import context.store as store
import util_python.param_util as pu

# example2=ctypes.CDLL('../../cmake/build/example2.cpython-36m-x86_64-linux-gnu.so')

from cmake.build.example2 import *
from context import context

from dist_gcn.models import GCN
#import autograd.autograd as atg
import autograd.autograd_new as autoG
from util_python import data_trans as dt


def printInfo(firstHopSetsForWorkers):
    print("local and remote vertex distribution:")
    print("local worker {0}:{1}".format(context.glContext.config['id'],
                                        len(firstHopSetsForWorkers[context.glContext.config['id']])))
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.config['id']:
            print("worker {0}:{1}".format(i, len(firstHopSetsForWorkers[i])))


def run_gnn(dgnnClient, model):
    # 从远程获取顶点信息（主要是边缘顶点一阶邻居信息）后，在本地进行传播
    # features, adjs, labels are based on the order of new id, and these only contains the local nodes
    #data = dt.load_data(dgnnClient)

    data = dt.load_datav2(dgnnClient)
    agg_node = data['agg_node']

    features = data['features']
    adjs = data['adjs']
    nodes_from_server = data['nodes_from_server']
    firstHopSetsForWorkers = data['firstHopSetsForWorkers']
    labels = data['labels']
    idx_val = data['idx_val']
    idx_train = data['idx_train']
    idx_test = data['idx_test']
    id_old2new_map = data['id_old2new_map']
    id_new2old_map = data['id_new2old_map']
    nodes = data['nodes'] # it just contains the local nodes
    train_ratio = data['train_ratio']
    test_ratio = data['test_ratio']
    val_ratio = data['val_ratio']
    train_num = data['train_num']
    test_num = data['test_num']
    val_num = data['val_num']

    #change add

    edges = []
    # 从adj中解析出edge
    # for i in range(len(adjs)):
    #     for nei_id in adjs[i]:
    #         edges.append([i, nei_id])
    # Yu
    for i in agg_node:
        for nei_id in adjs[i]:
            edges.append([i, nei_id])

    edges = np.array(edges)
    adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(len(id_old2new_map), len(id_old2new_map)),
                         dtype=np.int)

    # print(adjs.T)

    adjs = adjs + adjs.T.multiply(adjs.T > adjs) - adjs.multiply(adjs.T > adjs)
    adjs = dt.normalize_gcn(adjs + sp.eye(adjs.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    adjs = adjs[nodes]

    adjs = dt.sparse_mx_to_torch_sparse_tensor(adjs)  # 邻接矩阵转为tensor处理



    printInfo(firstHopSetsForWorkers)
    ifCompress = context.glContext.config['ifCompress']
    timeList=[]
    for epoch in range(context.glContext.config['IterNum']):
        startTimeTotle = time.time()
        model.train()
        # optimizer.zero_grad()
        store.isTrain = True
        if ifCompress:
            context.glContext.config['ifCompress'] = True

        # slow
        start = time.time()

        output = model(features, adjs, nodes_from_server, epoch) # change
        end = time.time()
        # print("output time:{0}".format(end - start))

        start_othertime = time.time()
        autograd.set_HZ(output, True, True, context.glContext.config["layerNum"])
        #change

        # if context.glContext.config["layerNum"] == 2:
        #     atg.H2 = output
        #     atg.Z2.required_grad = True
        #     atg.Z2.retain_grad()
        # elif context.glContext.config["layerNum"] == 3:
        #     atg.H3 = output
        #     atg.Z3.required_grad = True
        #     atg.Z3.retain_grad()
        # elif context.glContext.config["layerNum"] == 4:
        #     atg.H4 = output
        #     atg.Z4.required_grad = True
        #     atg.Z4.retain_grad()

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
        autograd.back_prop_detail(dgnnClient,model,id_new2old_map, nodes, epoch, adjs)
        # if context.glContext.config["layerNum"] == 2:
        #     atg.G2 = atg.Z2.grad
        #     # 算出G2后，给自己所在的server赋值
        #     G_list = atg.G2.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #
        #     dgnnClient.setG(G_dict, 2)
        #     # 同步一下
        #
        #
        #
        #     atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)
        #     atg.sigma_z1_grad = atg.Z1.data
        #     atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))
        #
        #     # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
        #     # 这段跟获取一阶邻居的嵌入表示类似,pull g2 to compute g1
        #     start = time.time()
        #
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 2)
        #     end = time.time()
        #     # if context.glContext.ifShowInfo:
        #     # print("pull neiborgh G time:{0}".format(end - start))
        #
        #     a = torch.spmm(adjs, atg.G2)
        #     b = torch.mm(a, model.gc2.weight.t())
        #     atg.G1 = torch.mul(b, atg.sigma_z1_grad)
        #     atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)
        #     atg.B1 = atg.G2.detach().numpy().sum(axis=0)
        #     atg.B0 = atg.G1.detach().numpy().sum(axis=0)
        #     np.set_printoptions(threshold=sys.maxsize)
        #
        #     model.gc1.weight.grad.data = atg.Y0
        #     model.gc1.bias.grad.data = torch.FloatTensor(atg.B0)
        #     model.gc2.weight.grad.data = atg.Y1
        #     model.gc2.bias.grad.data = torch.FloatTensor(atg.B1)
        #
        #
        # elif context.glContext.config["layerNum"] == 3:
        #     atg.G3 = atg.Z3.grad
        #     # 算出G2后，给自己所在的server赋值
        #     G_list = atg.G3.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #
        #     dgnnClient.setG(G_dict, 3)
        #     # 同步一下
        #
        #     # G3本地完全的，不需要从其他机器获取
        #     atg.Y2 = torch.mm(atg.A_X_H2.t(), atg.G3)
        #     atg.B2 = atg.G3.detach().numpy().sum(axis=0)
        #
        #     atg.sigma_z2_grad = atg.Z2.data
        #     atg.sigma_z2_grad = torch.tensor(np.where(atg.sigma_z2_grad > 0, 1, 0))
        #     # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
        #     # 这段跟获取一阶邻居的嵌入表示类似
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 3)
        #     a = torch.spmm(adjs, atg.G3)
        #     b = torch.mm(a, model.gc3.weight.t())
        #     atg.G2 = torch.mul(b, atg.sigma_z2_grad)
        #     atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)
        #     atg.B1 = atg.G2.detach().numpy().sum(axis=0)
        #
        #     # set G2
        #     G_list = atg.G2.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #     dgnnClient.setG(G_dict, 2)
        #
        #     atg.sigma_z1_grad = atg.Z1.data
        #     atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))
        #
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 2)
        #     a = torch.spmm(adjs, atg.G2)
        #     b = torch.mm(a, model.gc2.weight.t())
        #     atg.G1 = torch.mul(b, atg.sigma_z1_grad)
        #     atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)
        #     atg.B0 = atg.G1.detach().numpy().sum(axis=0)
        #
        #     np.set_printoptions(threshold=sys.maxsize)
        #     model.gc1.weight.grad.data = atg.Y0
        #     model.gc1.bias.grad.data = torch.FloatTensor(atg.B0)
        #     model.gc2.weight.grad.data = atg.Y1
        #     model.gc2.bias.grad.data = torch.FloatTensor(atg.B1)
        #     model.gc3.weight.grad.data = atg.Y2
        #     model.gc3.bias.grad.data = torch.FloatTensor(atg.B2)
        #
        # elif context.glContext.config["layerNum"] == 4:
        #     atg.G4 = atg.Z4.grad
        #
        #     # 算出G2后，给自己所在的server赋值
        #     G_list = atg.G4.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #
        #     dgnnClient.setG(G_dict, 4)
        #
        #     # G4本地完全的，不需要从其他机器获取
        #     atg.Y3 = torch.mm(atg.A_X_H3.t(), atg.G4)
        #     atg.B3 = atg.G4.detach().numpy().sum(axis=0)
        #
        #     atg.sigma_z3_grad = atg.Z3.data
        #     atg.sigma_z3_grad = torch.tensor(np.where(atg.sigma_z3_grad > 0, 1, 0))
        #     # 需要从其他机器中获取一阶邻居的G2，然后按照old-new重新拼接到G2中
        #     # 这段跟获取一阶邻居的嵌入表示类似
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 4)
        #     a = torch.spmm(adjs, atg.G4)
        #     b = torch.mm(a, model.gc4.weight.t())
        #     atg.G3 = torch.mul(b, atg.sigma_z3_grad)
        #     atg.Y2 = torch.mm(atg.A_X_H2.t(), atg.G3)
        #     atg.B2 = atg.G3.detach().numpy().sum(axis=0)
        #
        #     # set G3
        #     G_list = atg.G3.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #     dgnnClient.setG(G_dict, 3)
        #
        #     atg.sigma_z2_grad = atg.Z2.data
        #     atg.sigma_z2_grad = torch.tensor(np.where(atg.sigma_z2_grad > 0, 1, 0))
        #
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 3)
        #     a = torch.spmm(adjs, atg.G3)
        #     b = torch.mm(a, model.gc3.weight.t())
        #     atg.G2 = torch.mul(b, atg.sigma_z2_grad)
        #     atg.Y1 = torch.mm(atg.A_X_H1.t(), atg.G2)
        #     atg.B1 = atg.G2.detach().numpy().sum(axis=0)
        #
        #     # set G2
        #     G_list = atg.G2.detach().numpy().tolist()
        #     G_dict = {}
        #     for i in range(len(G_list)):
        #         G_dict[id_new2old_map[i]] = G_list[i]
        #     dgnnClient.setG(G_dict, 2)
        #
        #     atg.sigma_z1_grad = atg.Z1.data
        #     atg.sigma_z1_grad = torch.tensor(np.where(atg.sigma_z1_grad > 0, 1, 0))
        #
        #     context.glContext.dgnnServerRouter[0].server_Barrier(0)
        #     ra.pullNeighborG(nodes, epoch, 2)
        #     a = torch.spmm(adjs, atg.G2)
        #     b = torch.mm(a, model.gc2.weight.t())
        #     atg.G1 = torch.mul(b, atg.sigma_z1_grad)
        #     atg.Y0 = torch.mm(atg.A_X_H0.t(), atg.G1)
        #     atg.B0 = atg.G1.detach().numpy().sum(axis=0)
        #
        #     np.set_printoptions(threshold=sys.maxsize)
        #     model.gc1.weight.grad.data = atg.Y0
        #     model.gc1.bias.grad.data = torch.FloatTensor(atg.B0)
        #     model.gc2.weight.grad.data = atg.Y1
        #     model.gc2.bias.grad.data = torch.FloatTensor(atg.B1)
        #     model.gc3.weight.grad.data = atg.Y2
        #     model.gc3.bias.grad.data = torch.FloatTensor(atg.B2)
        #     model.gc4.weight.grad.data = atg.Y3
        #     model.gc4.bias.grad.data = torch.FloatTensor(atg.B3)

        # 求出梯度后，发送到参数服务器中进行聚合，并更新参数值
        # a=model.gc1.weight.grad
        # 将权重梯度和偏移梯度分别转化成map<int,vector<vector<float>>>和map<int,vector<float>>
        weights_grad_map = {}
        bias_grad_map = {}

        laynum = context.glContext.config["layerNum"]
        for i in range(laynum):
            weights_grad_map[i] = model.gc[i+1].weight.grad.detach().numpy().tolist()
            bias_grad_map[i] = model.gc[i+1].bias.grad.detach().numpy().tolist()
        # # 这里先不做通用的，直接假定是两层，之后再改
        # if context.glContext.config["layerNum"] == 2:
        #     weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
        #     bias_grad_map[0] = model.gc1.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[1] = model.gc2.bias.grad.detach().numpy().tolist()
        # elif context.glContext.config["layerNum"] == 3:
        #     weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[2] = model.gc3.weight.grad.detach().numpy().tolist()
        #     bias_grad_map[0] = model.gc1.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[1] = model.gc2.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[2] = model.gc3.bias.grad.detach().numpy().tolist()
        # elif context.glContext.config["layerNum"] == 4:
        #     weights_grad_map[0] = model.gc1.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[1] = model.gc2.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[2] = model.gc3.weight.grad.detach().numpy().tolist()
        #     weights_grad_map[3] = model.gc4.weight.grad.detach().numpy().tolist()
        #     bias_grad_map[0] = model.gc1.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[1] = model.gc2.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[2] = model.gc3.bias.grad.detach().numpy().tolist()
        #     bias_grad_map[3] = model.gc4.bias.grad.detach().numpy().tolist()
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

        if epoch % 5 == 0:
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

            print('Epoch: {:04d}'.format(epoch + 1),
                'acc_train_entire{:.4f}'.format(acc_entire['train'] / (float(train_datanum_entire))),
                  'acc_val_entire:{:.4f}'.format(acc_entire['val'] / (float(val_datanum_entire))),
                  'acc_test_entire: {:.4f}'.format(acc_entire['test'] / (float(test_datanum_entire))),
                  'f1_val_entire:{:.4f}'.format(acc_entire['val_f1'] / (float(val_datanum_entire))),
                  'f1_test_entire: {:.4f}'.format(acc_entire['test_f1'] / (float(test_datanum_entire))),
                  "epoch_iter_time: {0}".format(endTimeTotle - startTimeTotle),
                  "avg_iter_time: {0}".format(np.array(timeList).sum(axis=0)/len(timeList)))


    # context.glContext.config['ifCompress'] = False
    # store.isTrain = False
    # output = model(features, adjs, nodes_from_server, 10000)


if __name__ == "__main__":
    pp.parserInit()
    pp.printContext()
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

        autograd = autoG.AutoGrad()
        activation = [F.relu, F.relu, F.relu, F.relu]
        autograd.set_activation(activation)
        # assign parameter
        model = GCN(nfeat=context.glContext.config['feature_dim'],
                    nhid=context.glContext.config['hidden'],
                    nclass=context.glContext.config['class_num'],
                    dropout=0.5, autograd=autograd)

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
        run_gnn(context.glContext.dgnnClient, model)

    elif context.glContext.config['role'] == 'master':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)
        # ServiceImpl.RunServerByPy('127.0.0.1:4001')
