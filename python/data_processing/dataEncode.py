from ecgraph.context import context as ct
import numpy as np
import scipy.sparse as sp


fileReadName=ct.glContext.config['data_path']
fileWriteName=ct.glContext.config['data_path']
nodeNum=ct.glContext.config['data_num']
featDim=ct.glContext.config['feature_dim']
classNum=ct.glContext.config['class_num']


edgesFileReadName=fileReadName+'/edges_raw.txt'
featsFileReadName=fileReadName+'/featsClass_raw.txt'

edgesFileWriteName=fileWriteName+'/edges.txt'
featsFileWriteName=fileWriteName+'/featsClass.txt'


def feature_normalize(sparse_matrix):
    """Normalize sparse matrix feature by row.
    Reference:
      DGL(https://github.com/dmlc/dgl).
    """
    mx_abs=np.abs(sparse_matrix)
    row_sum = np.array(mx_abs.sum(1))
    row_norm = np.power(row_sum, -1).flatten()
    row_norm[np.isinf(row_norm)] = 0.
    row_matrix_norm = sp.diags(row_norm)
    sparse_matrix_ret = row_matrix_norm.dot(sparse_matrix)
    # norm=np.array(sparse_matrix.multiply(sparse_matrix).sum(1))
    # norm=np.power(norm,-1/2).flatten()
    # norm[np.isinf(norm)]=0.
    # matrix_norm = sp.diags(norm)
    # sparse_matrix_ret=matrix_norm.dot(np.abs(sparse_matrix))

    return sparse_matrix_ret

if __name__ == '__main__':
    featsFileRead=open(featsFileReadName,'r')
    allLines=featsFileRead.readlines()
    featsFileWrite=open(featsFileWriteName,'w+')

    nodeMapDict={}
    countNode=0
    countLabel=0
    labelDict={}
    for eachLine in allLines:
        featClassStr=''
        lineSplit=eachLine.split('\t')
        nodeMapDict[lineSplit[0]]=countNode
        featClassStr+=str(countNode)+'\t'
        countNode+=1
        feat=[]
        for dimId in range(1,featDim+1):
            feat.append(float(lineSplit[dimId]))

        # feat = feature_normalize(sp.csc_matrix(feat))
        feat=sp.csc_matrix(feat)
        feat = np.array(feat.todense())
        featClassStr+='\t'.join(map(str,feat[0]))

        class_str=lineSplit[featDim+1]
        class_str=class_str[:-1]
        class_int=-1
        if labelDict.__contains__(class_str):
            class_int=labelDict[class_str]
        else:
            class_int=countLabel
            labelDict[class_str]=countLabel
            countLabel+=1
        featClassStr+='\t'+str(class_int)+'\n'
        # featClassStr+=lineSplit[featDim+1]
        featsFileWrite.write(featClassStr)
    print('node numbers:{0}'.format(countNode))
    print('label numbers:{0}'.format(countLabel))
    featsFileRead.close()
    featsFileWrite.close()
    # start to encode adjacent list
    edgesFileRead=open(edgesFileReadName,'r')
    edgesFileWrite=open(edgesFileWriteName,'w+')
    allLines=edgesFileRead.readlines()
    edgesMap=set()
    edgeCount=0
    for eachLine in allLines:
        edgesStr=''
        edgesStrReverse=''
        lineSplit=eachLine.split('\t')
        lineSplit[1]=lineSplit[1].replace('\n','')
        if lineSplit[0]!=lineSplit[1]:
            edgesStr+=str(nodeMapDict[lineSplit[0]])+'\t'+str(nodeMapDict[lineSplit[1]])+'\n'
            edgesStrReverse+=str(nodeMapDict[lineSplit[1]])+'\t'+str(nodeMapDict[lineSplit[0]])+'\n'
            if (not edgesMap.__contains__(edgesStr)) and (not edgesMap.__contains__(edgesStrReverse)):
                edgeCount+=1
                edgesMap.add(edgesStr)
                edgesMap.add(edgesStrReverse)
            else:
                edgesMap.add(edgesStr)
                edgesMap.add(edgesStrReverse)
            edgesFileWrite.write(edgesStr)
    for id in range(countNode):
        edgeStr=str(id)+'\t'+str(id)+'\n'
        edgesFileWrite.write(edgeStr)
    print('edges number:{0}'.format(edgeCount))
    edgesFileWrite.close()

