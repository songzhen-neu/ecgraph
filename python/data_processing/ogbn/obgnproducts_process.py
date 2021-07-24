import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from torch_geometric.datasets import Reddit
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import sys, os, ctypes
import numpy as np

root = osp.join(osp.dirname(osp.realpath(__file__)), '../..', '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]


fpFeatClass = 'data/ogbn-products/featsClass.txt'
fpEdge = 'data/ogbn-products/edges.txt'

fwFeatClass = open(fpFeatClass, 'w+')
fwEdge = open(fpEdge, 'w+')

edges=np.array(data['edge_index'])

edgeNum=len(edges[0])
for i in range(edgeNum):
    edge_tmp=str(edges[0][i])+'\t'+str(edges[1][i])+'\n'
    fwEdge.write(edge_tmp)

nodeNum=len(data['x'])
feats=np.array(data['x'])
classes=np.array(data['y'])

for i in range(nodeNum):
    featClass_tmp=str(i)+'\t'+'\t'.join(map(str,feats[i]))+'\t'+str(classes[i][0])+'\n'
    fwFeatClass.write(featClass_tmp)

fwFeatClass.close()
fwEdge.close()



