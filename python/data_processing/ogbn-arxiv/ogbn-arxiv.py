from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import time
# root = "/mnt/data"
dataset = PygNodePropPredDataset( name="ogbn-arxiv")
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
data = dataset[0]

train_idx=np.array(train_idx)
valid_idx=np.array(valid_idx)
test_idx=np.array(test_idx)

idx=np.append(train_idx,valid_idx)
idx=np.append(idx,test_idx)

idx=set(idx)

fpFeatClass = 'featsClass.txt'
fpEdge = 'edges.txt'

fwFeatClass = open(fpFeatClass, 'w+')
fwEdge = open(fpEdge, 'w+')

edges=np.array(data['edge_index'])

start=time.time()
edgeNum=len(edges[0])
edge=''
for i in range(edgeNum):
    if idx.__contains__(int(edges[0][i])) and idx.__contains__(int(edges[1][i])):
        edge_tmp = str(edges[0][i]) + '\t' + str(edges[1][i]) + '\n'
        # edge += edge_tmp
        fwEdge.write(edge_tmp)

nodeNum=len(data['x'])
feats=np.array(data['x'])
classes=np.array(data['y'])
featclass=''
for i in range(nodeNum):
    if idx.__contains__(i):
        featClass_tmp=str(i)+'\t'+'\t'.join(map(str,feats[i]))+'\t'+str(classes[i][0])+'\n'
        # featclass+=featClass_tmp
        fwFeatClass.write(featClass_tmp)

fwFeatClass.close()
fwEdge.close()
end=time.time()

print("time:{0}".format(end-start))