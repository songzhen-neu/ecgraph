import torch.nn as nn
import torch.nn.functional as F
from example.dist_gcn_param.layers import GraphConvolution
from ecgraph.context import context
import torch
import ecgraph.util_python.remote_access as ra


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, autograd):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        torch.manual_seed(1)
        nhid_len = len(nhid)
        self.gc = []
        for i in range(1, nhid_len + 2):
            if i == 1:
                self.gc.append(GraphConvolution(nfeat, nhid[0], 0))
            elif i != nhid_len + 1:
                self.gc.append(GraphConvolution(nhid[i - 2], nhid[i - 1], i - 1))
            elif i == nhid_len + 1:
                self.gc.append(GraphConvolution(nhid[i - 2], nclass, i - 1))
        self.layerNum = len(nhid) + 1
        self.dropout = dropout
        self.autograd = autograd

    def forward(self, x, adj, nodes, epoch, graph):
        for layid in range(self.layerNum):
            x = self.gc[layid](x, adj, nodes, epoch, self.autograd, graph)
            if not layid == self.layerNum - 1:
                self.autograd.Z[layid + 1] = x
                x = F.relu(x)
                self.autograd.H[layid + 1] = x
            else:
                self.autograd.Z[layid + 1] = x

        # x = self.autograd.forward_detail(self, x, adj, nodes, epoch, graph)
        remoteDataNum = 0
        for i in range(len(graph.fsthop_for_worker)):
            if i != context.glContext.config['id']:
                remoteDataNum = remoteDataNum + len(graph.fsthop_for_worker[i])
        if context.glContext.config['isChangeBitNum']:
            ra.changeCompressBit(context.glContext.dgnnClientRouterForCpp.get_comp_percent(remoteDataNum,
                                                                                           context.glContext.config[
                                                                                               'layerNum']))
        self.autograd.softmax_value = F.softmax(x, dim=1)
        # print("bitNum:{0}".format(int(context.glContext.config['bitNum'])))
        return F.log_softmax(x, dim=1)
