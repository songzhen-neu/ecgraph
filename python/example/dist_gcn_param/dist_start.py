import sys, os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
# sys.path.insert(2, BASE_PATH + '/../../')
print(BASE_PATH)
import time
import numpy as np
from sklearn.metrics import f1_score
import util_python.param_parser as pp
import util_python.metric as metric
from sklearn.metrics import accuracy_score
import psutil
import ecgraph
# BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_PATH)
# # sys.path.insert(1, BASE_PATH + '/../')
import torch.nn.functional as F
import scipy.sparse as sp
import context.store as store
import util_python.param_util as pu
# import torch.optim as optim
import optimizer as optim
# from optimizer.adam import Adam

# example2=ctypes.CDLL('../../cmake/build/example2.cpython-36m-x86_64-linux-gnu.so')

from cmake.build.lib.pb11_ec import *
from context import context

from example.dist_gcn_agg_grad.models import GCN
import autograd.autograd_new as autoG
from util_python import data_trans as dt
import torch as torch

from multiprocessing import cpu_count


cpu_num = cpu_count()  # 自动获取最大核心数目
print("cpu num:{0}".format(cpu_num))
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def printInfo(firstHopSetsForWorkers):
    print("local and remote vertex distribution:")
    print("local worker {0}:{1}".format(context.glContext.config['id'],
                                        len(firstHopSetsForWorkers[context.glContext.config['id']])))
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.config['id']:
            print("worker {0}:{1}".format(i, len(firstHopSetsForWorkers[i])))


def get_adjs_train(agg_node, adjs, nodes, ll):
    edges = []
    for i in agg_node:
        for nei_id in adjs[i]:
            edges.append([i, nei_id])

    edges = np.array(edges)
    adjs_train = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                               shape=(ll, ll),
                               dtype=np.int)

    # print(adjs.T)

    adjs_train = adjs_train + adjs_train.T.multiply(adjs_train.T > adjs_train) - adjs_train.multiply(
        adjs_train.T > adjs_train)
    adjs_train = dt.normalize_gcn(adjs_train + sp.eye(adjs_train.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    adjs_train = adjs_train[nodes]

    adjs_train = dt.sparse_mx_to_torch_sparse_tensor(adjs_train)  # 邻接矩阵转为tensor处理

    return adjs_train


def getGlobalGrad():
    server_num = context.glContext.config['server_num']
    layer_num = context.glContext.config['layerNum']
    agg_grad = {}
    for key in context.glContext.gradients:
        num_g = int(len(context.glContext.gradients[key]) / server_num)
        agg_grad[key] = []
        for sid in range(server_num):
            if sid == server_num - 1:
                agg_grad[key].extend(
                    context.glContext.dgnnServerRouter[sid].server_aggGrad(context.glContext.config['id'], sid,
                                                                           context.glContext.config['lr'],
                                                                           key,
                                                                           context.glContext.gradients[key][
                                                                           sid * num_g:]))

            else:
                agg_grad[key].extend(
                    context.glContext.dgnnServerRouter[sid].server_aggGrad(context.glContext.config['id'], sid,
                                                                           context.glContext.config['lr'],
                                                                           key,
                                                                           context.glContext.gradients[key][
                                                                           sid * num_g:(sid + 1) * num_g]))

    feat_size = context.glContext.config['feature_dim']
    class_num = context.glContext.config['class_num']
    hidden = context.glContext.config['hidden']
    for i in range(layer_num):
        if i == 0:
            autograd.weight_g[i]=torch.FloatTensor(agg_grad['w' + str(i)]).reshape(feat_size, hidden[0])
            autograd.bias_g[i]=torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[0])
            autograd.grads[i*2]=torch.FloatTensor(agg_grad['w' + str(i)]).reshape(feat_size, hidden[0])
            autograd.grads[i*2+1]=torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[0])
            # model.gc[i].weight.grad.data = torch.FloatTensor(agg_grad['w' + str(i)]).reshape(feat_size, hidden[0])
            # model.gc[i].bias.grad.data = torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[0])
        elif i == layer_num - 1:
            autograd.weight_g[i]= torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[-1], class_num)
            autograd.bias_g[i]= torch.FloatTensor(agg_grad['b' + str(i)]).reshape(class_num)
            autograd.grads[i*2]=torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[-1], class_num)
            autograd.grads[i*2+1]=torch.FloatTensor(agg_grad['b' + str(i)]).reshape(class_num)
            # model.gc[i].weight.grad.data = torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[-1], class_num)
            # model.gc[i].bias.grad.data = torch.FloatTensor(agg_grad['b' + str(i)]).reshape(class_num)
        else:
            autograd.weight_g[i]= torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[i - 1], hidden[i])
            autograd.bias_g[i]= torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[i])
            autograd.grads[i*2]=torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[i - 1], hidden[i])
            autograd.grads[i*2+1]=torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[i])
            # model.gc[i].weight.grad.data = torch.FloatTensor(agg_grad['w' + str(i)]).reshape(hidden[i - 1], hidden[i])
            # model.gc[i].bias.grad.data = torch.FloatTensor(agg_grad['b' + str(i)]).reshape(hidden[i])


# def updateParam():


def clear_time():
    for id in context.glContext.time_epoch.keys():
        context.glContext.time_epoch[id]=0

def run_gnn(dgnnClient, model):
    # 从远程获取顶点信息（主要是边缘顶点一阶邻居信息）后，在本地进行传播
    # features, adjs, labels are based on the order of new id, and these only contains the local nodes
    data = dt.load_data(dgnnClient)

    idx_val = data['idx_val']
    idx_train = data['idx_train']
    idx_test = data['idx_test']
    idx_train_for_test = data['idx_train_for_test']
    train_ratio = data['train_ratio']
    test_ratio = data['test_ratio']
    val_ratio = data['val_ratio']
    train_num = data['train_num']
    test_num = data['test_num']
    val_num = data['val_num']
    graph_full = data['graph_full']
    # graph_train = data['graph_train']

    laynum = context.glContext.config["layerNum"]
    server_num = context.glContext.config['server_num']
    optimizer = optim.Adam(model.parameters(), lr=context.glContext.config['lr'], weight_decay=5e-4)

    edges = []
    # 从adj中解析出edge
    for i in range(len(graph_full.adj)):
        for nei_id in graph_full.adj[i]:
            edges.append([i, nei_id])
    edges = np.array(edges)
    adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(len(graph_full.id_old2new_dict), len(graph_full.id_old2new_dict)),
                         dtype=np.int)
    adjs = adjs + adjs.T.multiply(adjs.T > adjs) - adjs.multiply(adjs.T > adjs)
    adjs = dt.normalize_gcn(adjs + sp.eye(adjs.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    adjs = adjs[range(len(graph_full.train_vertices))]
    adjs = dt.sparse_mx_to_torch_sparse_tensor(adjs)  # 邻接矩阵转为tensor处理
    graph_full.adj = adjs
    printInfo(graph_full.fsthop_for_worker)

    ifCompress = context.glContext.config['ifCompress']
    timeList = []
    for epoch in range(context.glContext.config['iterNum']):
        startTimeTotle = time.time()
        optimizer.zero_grad()
        model.train()
        store.isTrain = True
        if ifCompress:
            context.glContext.config['ifCompress'] = True

        # slow
        start = time.time()
        output = model(graph_full.feat_data, graph_full.adj, graph_full.train_vertices, epoch, graph_full)  # change
        end = time.time()
        context.glContext.time_epoch['forward']+=(end-start)

        start_othertime = time.time()
        autograd.set_HZ(output, True, True, context.glContext.config["layerNum"])

        loss_train = F.nll_loss(output[idx_train], graph_full.label[idx_train])
        acc_train = accuracy_score(graph_full.label[idx_train].detach().numpy(),
                                   output[idx_train].detach().numpy().argmax(axis=1))

        start_backward = time.time()

        # loss_train.backward()  # 反向求导  Back Propagation

        grad_nll=ecgraph.BackWard.NllLossBackward(output,graph_full.label,idx_train)
        grad_softmax=ecgraph.BackWard.LogSoftmaxBackward(autograd.softmax_value,grad_nll)
        autograd.Z_grad[laynum]=grad_softmax

        end_backward = time.time()
        context.glContext.time_epoch['backward']+=(end_backward-start_backward)

        # 需要准确的反向传播过程
        start_bp_manu = time.time()
        autograd.back_prop_detail(dgnnClient, model, graph_full.id_new2old_dict, graph_full.train_vertices, epoch,
                                  graph_full.adj, graph_full)
        end_bp_manu = time.time()
        context.glContext.time_epoch['backward_m']+=(end_bp_manu-start_bp_manu)


        # print("backward time:{},{}".format(end_backward-start_backward,end_bp_manu-start_bp_manu))
        # 求出梯度后，发送到参数服务器中进行聚合，并更新参数值
        # a=model.gc1.weight.grad
        # 将权重梯度和偏移梯度分别转化成map<int,vector<vector<float>>>和map<int,vector<float>>


        start_update = time.time()
        for i in range(laynum):
            context.glContext.gradients['w' + str(i)]=autograd.Y[i].detach().flatten().numpy().tolist()
            context.glContext.gradients['b' + str(i)]=autograd.B[i].tolist()
            # context.glContext.gradients['w' + str(i)] = model.gc[i].weight.grad.detach().flatten().numpy().tolist()
            # context.glContext.gradients['b' + str(i)] = model.gc[i].bias.grad.detach().flatten().numpy().tolist()

        getGlobalGrad()
        optimizer.step()
        # updateParam()

        # 返回参数后同步,这里barrier的参数暂时没有用到
        context.glContext.dgnnServerRouter[0].server_Barrier(0)

        end_update = time.time()
        context.glContext.time_epoch['update']+=(end_update-start_update)

        endTimeTotle = time.time()

        timeList.append(endTimeTotle - startTimeTotle)

        model.eval()

        # output = model(features, adjs,nodes_from_server)

        if epoch % context.glContext.config['print_result_interval'] == 0:
            context.glContext.config['ifCompress'] = False
            store.isTrain = False
            output = model(graph_full.feat_data, graph_full.adj, graph_full.train_vertices, 10000, graph_full)
            loss_val = F.nll_loss(output[idx_val], graph_full.label[idx_val])  # 验证集的损失函数
            # acc_val = metric.accuracy(output[idx_val], labels[idx_val])
            acc_val = accuracy_score(graph_full.label[idx_val].detach().numpy(),
                                     output[idx_val].detach().numpy().argmax(axis=1))

            # acc_train = metric.accuracy(output[idx_train], labels[idx_train])  # 计算准确率
            loss_test = F.nll_loss(output[idx_test], graph_full.label[idx_test])  # 验证集的损失函数
            acc_test = metric.accuracy(output[idx_test], graph_full.label[idx_test])

            val_f1 = f1_score(graph_full.label[idx_val].detach().numpy(),
                              output[idx_val].detach().numpy().argmax(axis=1),
                              average='micro')

            test_f1 = f1_score(graph_full.label[idx_test].detach().numpy(),
                               output[idx_test].detach().numpy().argmax(axis=1),
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
                  "epoch_iter_time: {:.4f}".format(endTimeTotle - startTimeTotle-context.glContext.time_epoch['backward']),
                  "avg_iter_time: {:.4f}".format(np.array(timeList).sum(axis=0) / len(timeList)),
                  "set_embs_time:{:.4f}".format(context.glContext.time_epoch['set_embs']),
                  "get_embs_time:{:.4f}".format(context.glContext.time_epoch['get_embs']),
                  "forward:{:.4f}".format(context.glContext.time_epoch['forward']),
                  "backward:{:.4f}".format(context.glContext.time_epoch['backward']),
                  "backward_m:{:.4f}".format(context.glContext.time_epoch['backward_m']),
                  "update:{:.4f}".format(context.glContext.time_epoch['update']),
                  "get_g:{:.4f}".format(context.glContext.time_epoch['get_g']),
                  "set_g:{:.4f}".format(context.glContext.time_epoch['set_g']),
                  "memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        clear_time()

            # print('acc_test:{:.4f}'.format(acc_test))

    # context.glContext.config['ifCompress'] = False
    # store.isTrain = False
    # output = model(features, adjs, nodes_from_server, 10000)


if __name__ == "__main__":
    pp.parserInit()
    pp.printContext()

    if context.glContext.config['role'] == 'server':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['server_address'][context.glContext.config['id']],
                                  context.glContext.worker_id)

    elif context.glContext.config['role'] == 'worker':
        context.glContext.initCluster()

        # 初始化参数服务器模型，现在假设参数服务器就一台机器，先不进行参数划分
        # 输入：节点属性维度、隐藏层维度、标签维度

        autograd = autoG.autograd
        activation = [F.relu, F.relu, F.relu, F.relu, F.relu, F.relu]  # 第0个激活层没用到
        autograd.set_activation(activation)
        # assign parameter
        model = GCN(nfeat=context.glContext.config['feature_dim'],
                    nhid=context.glContext.config['hidden'],
                    nclass=context.glContext.config['class_num'],
                    dropout=0.5, autograd=autograd)

        pu.assignParam()
        context.glContext.dgnnClient.initCompressBitMap(context.glContext.config['bitNum'])
        context.glContext.dgnnServerRouter[0].server_Barrier(0)
        # if id == 0:
        #     context.glContext.dgnnMasterRouter.freeMaster()

        # # 已经将各自数据放到了数据库里，接下来定义GNN模型，然后训练
        # # 这里的操作的workerStore是全局静态变量，因此整个进程都可见
        run_gnn(context.glContext.dgnnClient, model)

    elif context.glContext.config['role'] == 'master':
        context.glContext.worker_id = context.glContext.config['id']
        ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)
