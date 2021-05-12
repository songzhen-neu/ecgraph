import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=5):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else set(to_neigh) for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs  # 采样 如果样本数大于邻居数则样本为邻居  type(samp_neighs): set(set([nodes]))
        # self.gcn=True
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 加当前节点的特征
        unique_nodes_list = list(set.union(*samp_neighs))  # 得到一个不重复的节点列表
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}  # 得到一个不重复的节点编号
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))  # 构建mask为 samp_neighs * unique_nodes 矩阵
        # build a new graph for training, however the row and column index are not referring to the same index codes.
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 列索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]  # 行索引
        mask[row_indices, column_indices] = 1  # 有边的位置设为1（邻接矩阵）
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)  # 邻居节点数集合 （一维）
        mask = mask.div(num_neigh)  # 平均边权重
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))  # 令嵌入矩阵为特征矩阵
        to_feats = mask.mm(embed_matrix)  # 聚合到的邻居特征矩阵
        return to_feats
