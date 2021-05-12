import math

import torch

import torch.nn as nn

from context import context
import numpy as np
import autograd.autograd as atg
import context.momentum as mom
import context.store as store
from multiprocessing import Process, Queue
# from cmake.build.example2 import *
from cmake.build.example2 import *
import _thread
from threading import Lock
import threading

import time


# from dist_gcn.dist_start import dgnnServerRouter


def accessEmb(needed_emb_map_from_workers, i, epoch, layer_id):
    if context.glContext.config['isChangeRate'] and store.isTrain:
        # print("epoch:{0}".format(epoch))
        needed_emb_map_from_workers[i] = \
            context.glContext.dgnnWorkerRouter[i].worker_pull_emb_trend(
                context.glContext.config['firstHopForWorkers'][i],
                layer_id, epoch,
                context.glContext.config['bucketNum'],
                context.glContext.worker_id, i,
                context.glContext.config['layerNum'],
                context.glContext.config['trend'],
                context.glContext.config['bitNum'])
        if (epoch + 1) % context.glContext.config['trend'] == 0:
            if not store.changeRate.__contains__(i):
                store.changeRate[i] = {}
                store.embs[i] = {}

            store.changeRate[i][layer_id] = \
                context.glContext.dgnnWorkerRouter[i].getChangeRate(i, layer_id)
            store.embs[i][layer_id] = needed_emb_map_from_workers[i]

        if int((epoch) / context.glContext.config['trend']) > 0 and (epoch + 1) % context.glContext.config[
            'trend'] != 0:
            round = (epoch + 1) % context.glContext.config['trend']
            changeEmb = store.embs[i][layer_id] + round * store.changeRate[i][layer_id]
            needed_emb_map_from_workers[i] = (needed_emb_map_from_workers[i] + changeEmb) / 2
    else:
        if context.glContext.config['ifCompress']:
            needed_emb_map_from_workers[i] = \
                context.glContext.dgnnWorkerRouter[i].worker_pull_emb_compress(
                    context.glContext.config['firstHopForWorkers'][i],
                    context.glContext.config['ifCompensate'], layer_id, epoch,
                    context.glContext.config['compensateMethod'],
                    context.glContext.config['bucketNum'],
                    context.glContext.config['changeToIter'],
                    context.glContext.worker_id,
                    context.glContext.config['layerNum'],
                    context.glContext.config['bitNum'])
        else:
            # needed_emb_map_from_workers[i] = \
            #     context.glContext.dgnnWorkerRouter[i].worker_pull_needed_emb_compress_iter(
            #     context.glContext.config['firstHopForWorkers'][i],
            #     context.glContext.config['ifCompensate'],self.layer_id,epoch,
            #     context.glContext.config['bucketNum']
            # )
            needed_emb_map_from_workers[i] = \
                context.glContext.dgnnWorkerRouter[i].worker_pull_needed_emb(
                    context.glContext.config['firstHopForWorkers'][i], epoch, layer_id, context.glContext.config['id'],
                    i)
    store.threadCountList[layer_id] += 1


class GraphConvolution(nn.Module):

    # 初始化层：输入feature维度，输出feature维度，权重，偏移
    def __init__(self, in_features, out_features, layer_id, bias=True):
        super(GraphConvolution, self).__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features),
                                   requires_grad=True)  # FloatTensor建立tensor
        # 常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        # 绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
            # Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
        # self.reset_parameters()

    # 初始化权重
    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
    #     self.weight.data_raw.uniform_(-stdv, stdv)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
    #     if self.bias is not None:
    #         self.bias.data_raw.uniform_(-stdv, stdv)

    '''
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''

    def forward(self, input, adj, nodes, epoch, weights, bias):
        # 权重需要从参数服务器中获取,先不做参数划分了，只弄一个server
        # 从参数服务器获取第0层的参数
        context.glContext.dgnnServerRouter[0].server_Barrier(self.layer_id)

        if not context.glContext.config['isNeededExactBackProp']:
            # 更新自身weights和bias
            self.weight.data = torch.FloatTensor(weights)
            self.bias.data = torch.FloatTensor(bias)
            support = None
            emb_temp = None
            if not context.glContext.config['firstProp']:
                # 在每个顶点做完神经网络变换后，再进行传播
                support = torch.mm(input, self.weight)
                # 将support设置到dgnnClient里,需要转成原index
                emb_temp = support.detach().numpy().tolist()
            else:
                support = input
                emb_temp = support.detach().numpy().tolist()

            emb_dict = {}
            for id in range(len(nodes)):
                emb_dict[nodes[id]] = emb_temp[id]
            context.glContext.dgnnClient.worker_setEmbs(emb_dict)

            # 变换后，需要从远端获取嵌入;也就是获取一阶邻居的嵌入；
            # 需要每台机器先同步一下，确定所有机器都做完了这步计算,在server上同步即可，传递层id
            context.glContext.dgnnServerRouter[0].server_Barrier(self.layer_id)

            # 去指定worker获取一阶邻居
            needed_emb_map_from_workers = {}

            start = time.time()
            for i in range(context.glContext.config['worker_num']):
                if i != context.glContext.worker_id:
                    if not context.glContext.config['isChangeRate']:
                        if context.glContext.config['ifCompress']:
                            needed_emb_map_from_workers[i] = \
                                context.glContext.dgnnWorkerRouter[i].worker_pull_emb_compress(
                                    context.glContext.config['firstHopForWorkers'][i],
                                    context.glContext.config['ifCompensate'], self.layer_id, epoch,
                                    context.glContext.config['compensateMethod'],
                                    context.glContext.config['bucketNum'],
                                    context.glContext.config['changeToIter'],
                                    context.glContext.worker_id,
                                    context.glContext.config['layerNum'])
                        else:
                            needed_emb_map_from_workers[i] = \
                                context.glContext.dgnnWorkerRouter[i].worker_pull_needed_emb(
                                    context.glContext.config['firstHopForWorkers'][i], epoch, self.layer_id)
                    else:
                        needed_emb_map_from_workers[i] = \
                            context.glContext.dgnnWorkerRouter[i].worker_pull_emb_trend(
                                context.glContext.config['firstHopForWorkers'][i],
                                self.layer_id, epoch,
                                context.glContext.config['bucketNum'],
                                context.glContext.worker_id,
                                context.glContext.config['layerNum'],
                                context.glContext.config['trend'])
                        if (epoch + 1) % context.glContext.config['trend'] == 0:
                            store.changeRate[i][self.layer_id] = \
                                context.glContext.dgnnWorkerRouter[i].getChangeRate(self.layer_id)
                            store.embs[i][self.layer_id] = needed_emb_map_from_workers[i]

                        if (epoch + 1) / context.glContext.config['trend'] > 0 and (epoch + 1) % \
                                context.glContext.config['trend'] != 0:
                            round = epoch % context.glContext.config['trend']
                            changeEmb = store.embs[i][self.layer_id] + round * store.changeRate[i][self.layer_id]
                            needed_emb_map_from_workers[i] = (needed_emb_map_from_workers[i] + changeEmb) / 2

            end = time.time()

            # print("time:{0}".format(end-start))
            # 获取完之后，将嵌入进行合并，按照新顶点的顺序
            # 这里其实是把一阶邻居的顶点的中间嵌入按照new id的顺序排列好
            needed_embs = [None] * (len(context.glContext.newToOldMap) - len(nodes))
            # for循环遍历每个从远端获取的特征
            for wid in range(context.glContext.config['worker_num']):
                if wid != context.glContext.worker_id:
                    for nid in context.glContext.config['firstHopForWorkers'][wid]:
                        new_id = context.glContext.oldToNewMap[nid] - len(nodes)
                        needed_embs[new_id] = needed_emb_map_from_workers[wid][nid]

            # for id in range(len(context.glContext.newToOldMap)):
            #     old_id=context.glContext.newToOldMap[id]
            #     worker_id=old_id%context.glContext.config['worker_num']
            #     emb_vec=[]
            #     if worker_id != context.glContext.worker_id:
            #         needed_embs[id-len(nodes)]=needed_emb_map_from_workers[worker_id]

            # 将needed_embs转化为tensor
            needed_embs = np.array(needed_embs)
            needed_embs = torch.FloatTensor(needed_embs)

            # print("support size:")
            # print(support.data_raw.shape)
            # print("needed embs size:")
            # print(needed_embs.shape)

            support = torch.cat((support, needed_embs), 0)

            # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
            # 稀疏矩阵乘法
            output = None
            if not context.glContext.config['firstProp']:
                output = torch.spmm(adj, support)
            else:
                output = torch.spmm(adj, support)
                output = torch.mm(output, self.weight)

            if self.bias is not None:
                return output + self.bias
            else:
                return output
        else:
            # 更新自身weights和bias
            self.weight.data = torch.FloatTensor(weights)
            self.bias.data = torch.FloatTensor(bias)

            # if epoch!=100 and self.layer_id==0:
            #     print('e{0}: l{1} weight_data:{2}'.format(epoch,self.layer_id,self.weight.data_raw[1][0]))
            # if epoch==2000:
            #     print('e{0}: l{1} weight_data:{2}'.format(epoch,self.layer_id,self.weight.data_raw))
            # print('e{0} l{1} bias_data:{2}'.format(epoch,self.layer_id,self.bias.data_raw[0]))

            # 将support设置到dgnnClient里,需要转成原index,slow
            emb_temp = input.detach().numpy()
            emb_dict = {}
            emb_nodes = np.array(nodes)
            # emb_feat=np.array(emb_temp)
            # for id in range(len(nodes)):
            #     # emb_dict[nodes[id]] = emb_temp[id]
            #     emb_nodes[id]=nodes[id]
            #     emb_feat[id]=emb_temp[id]

            # context.glContext.dgnnClient.worker_setEmbs(emb_dict)
            # context.glContext.dgnnClient.embs=emb_dict
            start = time.time()
            set_embs(emb_nodes, emb_temp)
            end = time.time()
            # print("set_embs time {0}:".format( end - start))
            # 变换后，需要从远端获取嵌入;也就是获取一阶邻居的嵌入；
            # 需要每台机器先同步一下，确定所有机器都做完了这步计算,在server上同步即可，传递层id,slow
            context.glContext.dgnnServerRouter[0].server_Barrier(self.layer_id)

            # temp=np.ones(50)

            # 去指定worker获取一阶邻居
            start = time.time()

            # firstHopForWorkersDict={}
            # for key in context.glContext.config['firstHopForWorkers']:
            #     firstHopForWorkersDict[key]=context.glContext.config['firstHopForWorkers'][key].tolist()
            feat_size = len(emb_temp[0])

            needed_embs = context.glContext.dgnnClientRouterForCpp.getNeededEmb(
                context.glContext.config['firstHopForWorkers'], epoch, self.layer_id, context.glContext.config['id'],
                context.glContext.oldToNewMap, context.glContext.config['worker_num'], len(nodes),
                context.glContext.config['ifCompress'], context.glContext.config['layerNum'],
                context.glContext.config['bitNum'], context.glContext.config['isChangeRate'], store.isTrain,
                context.glContext.config['trend'], feat_size)



            end = time.time()



            # 获取完之后，将嵌入进行合并，按照新顶点的顺序
            # 这里其实是把一阶邻居的顶点的中间嵌入按照new id的顺序排列好
            # needed_embs = [None] * (len(context.glContext.newToOldMap) - len(nodes))

            start = time.time()

            # for循环遍历每个从远端获取的特征
            # for wid in range(context.glContext.config['worker_num']):
            #     if wid != context.glContext.worker_id:
            #         for i, nid in enumerate(context.glContext.config['firstHopForWorkers'][wid]):
            #             new_id = context.glContext.oldToNewMap[nid] - len(nodes)
            #             needed_embs[new_id] = needed_emb_map_from_workers[wid][i]

            # needed_embs = np.array(needed_embs)

            if context.glContext.config['ifMomentum'] and epoch != 10000:
                if epoch == 0:
                    mom.hLast[self.layer_id] = needed_embs
                elif epoch == 1:
                    mom.mv1[self.layer_id] = needed_embs - mom.hLast[self.layer_id]
                    mom.hLast[self.layer_id] = needed_embs
                else:
                    mom.mv2[self.layer_id] = needed_embs - mom.hLast[self.layer_id]
                    mom.hLast[self.layer_id] = needed_embs
                    mv1l = mom.mv1[self.layer_id]
                    mv2l = mom.mv2[self.layer_id]
                    mv_mul = mv1l * mv2l
                    mv = np.where(mv_mul > 0, mv_mul, 0)
                    needed_embs = needed_embs + 1000 * mv

            # for id in range(len(context.glContext.newToOldMap)):
            #     old_id=context.glContext.newToOldMap[id]
            #     worker_id=old_id%context.glContext.config['worker_num']
            #     emb_vec=[]
            #     if worker_id != context.glContext.worker_id:
            #         needed_embs[id-len(nodes)]=needed_emb_map_from_workers[worker_id]

            # 将needed_embs转化为tensor

            needed_embs = torch.FloatTensor(needed_embs)
            # if self.layer_id==1:
            #     if epoch ==0:
            #         mom.hijlast=needed_embs[1][1]
            #     else:
            #         print(needed_embs[1][1]-mom.hijlast)
            #         mom.hijlast=needed_embs[1][1]

            # print("support size:")
            # print(support.data_raw.shape)
            # print("needed embs size:")
            # print(needed_embs.shape)

            input = torch.cat((input, needed_embs), 0)

            end = time.time()
            # print("concat time :{0}".format(end - start))

            if context.Context.config['isHCompensate'] and self.layer_id == 1:
                if epoch == 0:
                    atg.errorH1 = np.random.random(input.shape)
                    input = input - torch.FloatTensor(atg.errorH1)


                else:
                    input = input + torch.FloatTensor(atg.errorH1)
                    atg.errorH1 = np.random.random(input.shape)
                    input = input - torch.FloatTensor(atg.errorH1)

            # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
            # 稀疏矩阵乘法
            # aggregate = torch.spmm(adj, input)
            start = time.time()
            aggregate = torch.mm(adj, input)
            end = time.time()
            # print("aggregate time:{0}".format(end-start))

            if context.Context.config['isAggCompensate'] and self.layer_id == 1:
                if epoch == 0:
                    atg.errorAgg1 = np.random.random(aggregate.shape)
                    aggregate = aggregate - torch.FloatTensor(atg.errorAgg1)
                else:
                    aggregate = aggregate + torch.FloatTensor(atg.errorAgg1)
                    atg.errorAgg1 = np.random.random(aggregate.shape)
                    aggregate = aggregate - torch.FloatTensor(atg.errorAgg1)

            if self.layer_id == 0:
                atg.A_X_H0 = aggregate
            if self.layer_id == 1:
                atg.A_X_H1 = aggregate
            if self.layer_id == 2:
                atg.A_X_H2 = aggregate
            if self.layer_id == 3:
                atg.A_X_H3 = aggregate
            # 在每个顶点做完神经网络变换后，再进行传播

            start = time.time()
            output = torch.mm(aggregate, self.weight)
            end = time.time()
            # print("neural transform time:{0}".format(end-start))

            if self.bias is not None:
                return output + self.bias
            else:
                return output

    # 通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，#0.01，0.94]，
    # 里面的值代表该x对应标签不同的概率，故此值可转换为#[0,0,0,0,0,0,1]，对应我们之前把标签onehot后的第七种标签

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
