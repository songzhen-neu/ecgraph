import torch.nn as nn
import torch.nn.functional as F
from dist_gcn.layers import GraphConvolution
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
        self.gc.append(None)
        for i in range(1, nhid_len+2):
            if i == 1:
                self.gc.append(GraphConvolution(nfeat, nhid[0], 0))
                context.glContext.weights[0]=init.xavier_normal_(torch.FloatTensor(nfeat,nhid[0]))
                context.glContext.bias[0]=init.xavier_normal_(torch.FloatTensor(1,nhid[0]))
            elif i!=nhid_len+1:
                self.gc.append(GraphConvolution(nhid[i-2], nhid[i-1], i-1))
                context.glContext.weights[i-1]=init.xavier_normal_(torch.FloatTensor(nhid[i-2],nhid[i-1]))
                context.glContext.bias[i-1]=init.xavier_normal_(torch.FloatTensor(1,nhid[i-1]))
            elif i == nhid_len+1:
                self.gc.append(GraphConvolution(nhid[i-2], nclass, i-1))
                context.glContext.weights[i-1]=init.xavier_normal_(torch.FloatTensor(nhid[i-2],nclass))
                context.glContext.bias[i-1]=init.xavier_normal_(torch.FloatTensor(1,nclass))

        self.dropout = dropout
        self.autograd = autograd

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj, nodes, epoch):
        start = time.time()
        # add
        laynum = context.glContext.config['layerNum']
        weight = [[] for i in range(laynum)]
        bias = [[] for i in range(laynum)]

        for i in range(context.glContext.config['server_num']):
            for j in range(laynum):
                weight[j].extend(context.glContext.dgnnServerRouter[i].server_PullWeights(j))

        for i in range(laynum):
            bias[i] = context.glContext.dgnnServerRouter[0].server_PullBias(i)


        end = time.time()
        # print("pull weight time:{0}".format(end - start))

        x = self.autograd.forward_detail(self, x, adj,nodes, epoch, weight, bias)



        remoteDataNum=0
        for i in range(len(context.glContext.config['firstHopForWorkers'])):
            if i != context.glContext.config['id']:
                remoteDataNum=remoteDataNum+len(context.glContext.config['firstHopForWorkers'][i])
        if context.glContext.config['isChangeBitNum']:
            ra.changeCompressBit(context.glContext.dgnnClientRouterForCpp.get_comp_percent(remoteDataNum,context.glContext.config['layerNum']))

        # print("bitNum:{0}".format(int(context.glContext.config['bitNum'])))
        return F.log_softmax(x, dim=1)
