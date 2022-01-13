from ecgraph.context import context as ct

fileName = ct.glContext.config['data_path']
workerNum = ct.glContext.config['worker_num']
dataNum=ct.glContext.config['data_num']

metisFileName = fileName + '/metis.txt.part.' + str(workerNum)
fileWriteName = fileName + '/nodesPartition'+'.metis'+str(workerNum)+'.txt'

if __name__ == '__main__':
    if workerNum==1:
        metisFileWrite = open(fileWriteName, 'w+')
        for i in range(dataNum):
            metisFileWrite.write(str(i)+'\t')
        metisFileWrite.close()

    else:
        metisFileRead = open(metisFileName, 'r')
        metisFileWrite = open(fileWriteName, 'w+')
        allLines = metisFileRead.readlines()
        nodes = [str() for i in range(workerNum)]
        nodeId = 0
        for eachLine in allLines:
            eachLine = eachLine[:-1]
            workerId = int(eachLine)
            nodes[workerId] += str(nodeId) + '\t'
            nodeId += 1
        for i in range(workerNum):
            nodes[i] = nodes[i][:-1] + '\n'
            metisFileWrite.write(nodes[i])
        metisFileWrite.close()
        metisFileRead.close()

