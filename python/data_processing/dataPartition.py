
# fileName='../../data/test'
# nodeNum=20
# featDim=5
# classNum=2

import context.context as context

fileName=context.glContext.config['data_path']
nodeNum=context.glContext.config['data_num']
featDim=context.glContext.config['feature_dim']
classNum=context.glContext.config['class_num']

workerNum=context.glContext.config['worker_num']

fileWriteName=fileName+'/nodesPartition'+'.hash'+str(workerNum)+'.txt'
# hash partition
# nodes=[set()]*workerNum
# nodes=[set() for i in range(workerNum)]
nodes=[str() for i in range(workerNum)]
fileWrite=open(fileWriteName,'w+')
for i in range(nodeNum):
    # nodes[i%workerNum].add(i)
    nodes[i%workerNum]+=str(i)+'\t'
for i in range(workerNum):
    nodes[i]=nodes[i][:-1]
    nodes[i]+='\n'
    fileWrite.write(nodes[i])
fileWrite.close()

