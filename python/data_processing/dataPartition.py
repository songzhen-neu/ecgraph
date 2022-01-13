
# fileName='../../data/test'
# nodeNum=20
# featDim=5
# classNum=2

from ecgraph.context import context as context
import time

fileName= context.glContext.config['data_path']
nodeNum= context.glContext.config['data_num']
featDim= context.glContext.config['feature_dim']
classNum= context.glContext.config['class_num']

workerNum= context.glContext.config['worker_num']

fileWriteName=fileName+'/nodesPartition'+'.hash'+str(workerNum)+'.txt'
start=time.time()
# hash partition
# nodes=[set()]*workerNum
# nodes=[set() for i in range(workerNum)]
nodes=[str() for i in range(workerNum)]
fileWrite=open(fileWriteName,'w+')

for wid in range(workerNum):
    nodeAdd=str()
    for i in range(nodeNum):
        if i%workerNum==wid:
        # nodes[i%workerNum].add(i)
            nodeAdd+=str(i)+'\t'
    nodes[wid]=nodeAdd

for i in range(workerNum):
    nodes[i]=nodes[i][:-1]
    nodes[i]+='\n'
    fileWrite.write(nodes[i])
fileWrite.close()

end=time.time()
print("time:{0}".format(end-start))

