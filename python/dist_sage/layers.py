import torch

import torch.nn as nn

from context import context
import numpy as np
#import autograd.autograd as atg
import context.momentum as mom
import context.store as store
# from cmake.build.example2 import *
from cmake.build.example2 import *

import time






class SAGE_Layer(nn.Module):

    # 初始化层：输入feature维度，输出feature维度，权重，偏移
    def __init__(self, in_features, out_features, layer_id, bias=True):
        super(SAGE_Layer, self).__init__()
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
    #     ;if self.bias is not None:
    #         self.bias.data_raw.uniform_(-stdv, stdv)

    '''
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''

    def forward(self, input, adj, nodes, epoch, weights, bias, autograd):
        # 权重需要从参数服务器中获取,先不做参数划分了，只弄一个server
        # 从参数服务器获取第0层的参数
        context.glContext.dgnnServerRouter[0].server_Barrier(self.layer_id)

        # max1=torch.max(input)
        # min1=torch.min(input)
        # print("max:{0},min{1}:".format(max1,min1))
        # 更新自身weights和bias
        self.weight.data = torch.FloatTensor(weights)
        self.bias.data = torch.FloatTensor(bias)

        # 将support设置到dgnnClient里,需要转成原index,slow
        emb_temp = input.detach().numpy()
        emb_dict = {}
        emb_nodes = np.array(nodes)

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
            int(context.glContext.config['bitNum']), context.glContext.config['isChangeRate'], store.isTrain,
            context.glContext.config['trend'], feat_size,context.glContext.config['changeRateMode'])

        end = time.time()


        # 获取完之后，将嵌入进行合并，按照新顶点的顺序
        # 这里其实是把一阶邻居的顶点的中间嵌入按照new id的顺序排列好
        # needed_embs = [None] * (len(context.glContext.newToOldMap) - len(nodes))

        start = time.time()



        # 将needed_embs转化为tensor

        needed_embs = torch.FloatTensor(needed_embs)


        input = torch.cat((input, needed_embs), 0)

        end = time.time()
        # print("concat time :{0}".format(end - start))



        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        # 稀疏矩阵乘法
        # aggregate = torch.spmm(adj, input)
        start = time.time()
        aggregate = torch.mm(adj, input)
        end = time.time()
        # print("aggregate time:{0}".format(end-start))

        autograd.A_X_H[self.layer_id] = aggregate
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
