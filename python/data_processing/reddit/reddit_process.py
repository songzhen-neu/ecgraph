import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.datasets import Reddit
import sys, os, ctypes
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH + '/../data/Reddit')
dataset = Reddit(BASE_PATH + '/../data/Reddit')
data = dataset[0]

fpFeatClass = 'data/reddit/featsClass.txt'
fpEdge = 'data/reddit/edges.txt'

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



