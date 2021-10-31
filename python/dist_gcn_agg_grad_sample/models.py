import torch.nn as nn
import torch.nn.functional as F
from dist_gcn_agg_grad_sample.layers import GraphConvolution
from context import context
import time
from torch.nn import init
import torch
import util_python.remote_access as ra


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, autograd):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        torch.manual_seed(1)
        nhid_len = len(nhid)
        self.gc = []
        self.gc1 = None
        self.gc2 = None
        self.gc3 = None
        self.gc4 = None
        for i in range(1, nhid_len + 2):
            if i == 1:
                self.gc.append(GraphConvolution(nfeat, nhid[0], 0))
            elif i != nhid_len + 1:
                self.gc.append(GraphConvolution(nhid[i - 2], nhid[i - 1], i - 1))
            elif i == nhid_len + 1:
                self.gc.append(GraphConvolution(nhid[i - 2], nclass, i - 1))

        for i in range(len(self.gc)):
            if i == 0:
                self.gc1 = self.gc[i]
            elif i == 1:
                self.gc2 = self.gc[i]
            elif i == 2:
                self.gc3 = self.gc[i]
            elif i == 3:
                self.gc4 = self.gc[i]

        self.dropout = dropout
        self.autograd = autograd

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj, nodes, epoch, graph):
        x = self.autograd.forward_detail(self, x, adj, nodes, epoch, graph)
        remoteDataNum = 0
        for i in range(len(graph.fsthop_for_worker)):
            if i != context.glContext.config['id']:
                remoteDataNum = remoteDataNum + len(graph.fsthop_for_worker[i])
        if context.glContext.config['isChangeBitNum']:
            ra.changeCompressBit(context.glContext.dgnnClientRouterForCpp.get_comp_percent(remoteDataNum,
                                                                                           context.glContext.config[
                                                                                               'layerNum']))
        self.autograd.softmax_value=F.softmax(x,dim=1)
        # print("bitNum:{0}".format(int(context.glContext.config['bitNum'])))
        return F.log_softmax(x, dim=1)
