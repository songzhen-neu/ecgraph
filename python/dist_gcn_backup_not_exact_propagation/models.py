import torch.nn as nn
import torch
import torch.nn.functional as F
from dist_gcn.layers import GraphConvolution
from context import context
import autograd.autograd as atg
import time
from torch.nn import init
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        torch.manual_seed(1)
        if len(nhid) == 1:
            self.gc1 = GraphConvolution(nfeat, nhid[0], 0)  # gc1输入尺寸nfeat，输出尺寸nhid
            self.gc2 = GraphConvolution(nhid[0], nclass, 1)  # gc2输入尺寸nhid，输出尺寸nclass
            context.glContext.weights[0]=init.xavier_normal_(torch.FloatTensor(nfeat,nhid[0]))
            context.glContext.weights[1]=init.xavier_normal_(torch.FloatTensor(nhid[0],nclass))
            context.glContext.bias[0]=init.xavier_normal_(torch.FloatTensor(1,nhid[0]))
            context.glContext.bias[1]=init.xavier_normal_(torch.FloatTensor(1,nclass))

        elif len(nhid) == 2:
            self.gc1 = GraphConvolution(nfeat, nhid[0], 0)  # gc1输入尺寸nfeat，输出尺寸nhid
            self.gc2 = GraphConvolution(nhid[0], nhid[1], 1)
            self.gc3 = GraphConvolution(nhid[1], nclass, 2)
            context.glContext.weights[0]=init.xavier_normal_(torch.FloatTensor(nfeat,nhid[0]))
            context.glContext.weights[1]=init.xavier_normal_(torch.FloatTensor(nhid[0],nhid[1]))
            context.glContext.weights[2]=init.xavier_normal_(torch.FloatTensor(nhid[1],nclass))
            context.glContext.bias[0]=init.xavier_normal_(torch.FloatTensor(1,nhid[0]))
            context.glContext.bias[1]=init.xavier_normal_(torch.FloatTensor(1,nhid[1]))
            context.glContext.bias[2]=init.xavier_normal_(torch.FloatTensor(1,nclass))
        elif len(nhid)==3:
            self.gc1 = GraphConvolution(nfeat, nhid[0], 0)  # gc1输入尺寸nfeat，输出尺寸nhid
            self.gc2 = GraphConvolution(nhid[0], nhid[1], 1)
            self.gc3 = GraphConvolution(nhid[1], nhid[2], 2)
            self.gc4 = GraphConvolution(nhid[2], nclass, 3)
            context.glContext.weights[0]=init.xavier_normal_(torch.FloatTensor(nfeat,nhid[0]))
            context.glContext.weights[1]=init.xavier_normal_(torch.FloatTensor(nhid[0],nhid[1]))
            context.glContext.weights[2]=init.xavier_normal_(torch.FloatTensor(nhid[1],nhid[2]))
            context.glContext.weights[3]=init.xavier_normal_(torch.FloatTensor(nhid[2],nclass))
            context.glContext.bias[0]=init.xavier_normal_(torch.FloatTensor(1,nhid[0]))
            context.glContext.bias[1]=init.xavier_normal_(torch.FloatTensor(1,nhid[1]))
            context.glContext.bias[2]=init.xavier_normal_(torch.FloatTensor(1,nhid[2]))
            context.glContext.bias[3]=init.xavier_normal_(torch.FloatTensor(1,nclass))

        self.dropout = dropout

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj, nodes, epoch):
        start = time.time()
        weight0 =[]
        weight1 = []
        for i in range(context.glContext.config['server_num']):
            weight0.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(0))
            weight1.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(1))
        bia0 = context.glContext.dgnnServerRouter[0].server_PullBias(0)
        bia1 = context.glContext.dgnnServerRouter[0].server_PullBias(1)

        weight2 = []
        bia2 = []
        if context.glContext.config['layerNum'] == 3:
            for i in range(context.glContext.config['server_num']):
                weight2.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(2))
            bia2 = context.glContext.dgnnServerRouter[0].server_PullBias(2)

        weight3 = []
        bia3 = []
        if context.glContext.config['layerNum'] == 4:
            for i in range(context.glContext.config['server_num']):
                weight2.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(2))
                weight3.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(3))
            bia2 = context.glContext.dgnnServerRouter[0].server_PullBias(2)
            bia3 = context.glContext.dgnnServerRouter[0].server_PullBias(3)
        end = time.time()
        # print("pull weight time:{0}".format(end - start))

        if context.glContext.config['isNeededExactBackProp']:
            if context.glContext.config["layerNum"] == 2:
                x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc1(x, adj, nodes, epoch, weight0, bia0)
                x = F.normalize(x, p=2, dim=1)
                # print(x)
                # max1=torch.max(x)
                # x=torch.div(x,max1)
                atg.Z1 = x
                # print(x)
                x = F.relu(x)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
                # print(x)
                atg.H1 = x

                x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc2(x, adj, nodes, epoch, weight1, bia1)
                atg.Z2 = x
            elif context.glContext.config["layerNum"] == 3:
                x = self.gc1(x, adj, nodes, epoch, weight0, bia0)
                x = F.normalize(x, p=2, dim=1)

                atg.Z1 = x
                x = F.relu(x)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
                atg.H1 = x
                # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc2(x, adj, nodes, epoch, weight1, bia1)
                x = F.normalize(x, p=2, dim=1)

                atg.Z2 = x
                x = F.relu(x)
                atg.H2 = x
                x = self.gc3(x, adj, nodes, epoch, weight2, bia2)
                atg.Z3 = x
            elif context.glContext.config["layerNum"] == 4:
                x = self.gc1(x, adj, nodes, epoch, weight0, bia0)
                x = F.normalize(x, p=2, dim=1)
                atg.Z1 = x
                x = F.relu(x)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
                atg.H1 = x

                # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc2(x, adj, nodes, epoch, weight1, bia1)
                x = F.normalize(x, p=2, dim=1)
                atg.Z2 = x
                x = F.relu(x)
                atg.H2 = x

                # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc3(x, adj, nodes, epoch, weight2, bia2)
                x = F.normalize(x, p=2, dim=1)
                atg.Z3 = x
                x = F.relu(x)
                atg.H3 = x

                x = self.gc4(x, adj, nodes, epoch, weight3, bia3)
                atg.Z4 = x


        else:
            if context.glContext.config["layerNum"] == 2:
                x = self.gc1(x, adj, nodes, epoch, weight0, bia0)
                x = F.relu(x)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
                # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc2(x, adj, nodes, epoch, weight1, bia1)
            elif context.glContext.config["layerNum"] == 3:
                x = self.gc1(x, adj, nodes, epoch, weight0, bia0)
                x = F.relu(x)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
                # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
                x = self.gc2(x, adj, nodes, epoch, weight1, bia1)
                x = F.relu(x)
                x = self.gc3(x, adj, nodes, epoch, weight2, bia2)

        return F.log_softmax(x, dim=1)
