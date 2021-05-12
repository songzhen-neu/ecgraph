import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import context.context as context


# nn.Module is a callable class, so we can call the __call__ function by nn.Module()
# or by the enc(), where enc is a object of Encoder which extends the class 'nn.Module'


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=1000,layer_id=-1,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight=[]
        # self.weight = nn.Parameter(
        #         torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))     #构建权重矩阵
        torch.manual_seed(1)
        # init.xavier_uniform_(self.weight)    # 以均匀分布初始化
        self.layer_id = layer_id
        # self.weight.data_raw=torch.ones_like(self.weight.data_raw)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        for i in range(context.glContext.config['server_num']):
            self.weight.extend(context.glContext.dgnnServerRouter[i].server_PullWeights(self.layer_id))
        self.weight=torch.FloatTensor(self.weight)


        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                self.num_sample)          # 聚合到的邻居特征矩阵
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)      # if gcn:将邻居特征和自身特征拼接
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.t().mm(combined.t()))     # 使用relu激活函数得到输出
        return combined

