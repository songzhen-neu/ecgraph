from ecgraph.context import context as ct

fileName=ct.glContext.config['data_path']
nodeNum=ct.glContext.config['data_num']
featDim=ct.glContext.config['feature_dim']
classNum=ct.glContext.config['class_num']
edgesNum=ct.glContext.config['edge_num']


# fileName='../../data/cora'
# nodeNum=2708
# featDim=1433
# classNum=7
# edgesNum=5278

edgesFileName=fileName+'/edges.txt'
edgesMetisFileName=fileName+'/metis.txt'



if __name__ == '__main__':
    edgesFileRead=open(edgesFileName,'r')

    allLines=edgesFileRead.readlines()
    adjs={}
    count=0

    for eachLine in allLines:
        lineSplit=eachLine.split('\t')
        lineSplit[-1]=lineSplit[-1][:-1]
        vid=int(lineSplit[0])
        nid=int(lineSplit[1])
        if vid!=nid:
            if adjs.__contains__(vid):
                if not adjs[vid].__contains__(nid):
                    adjs[vid].add(nid)
                    count+=1
            else:
                set_tmp=set()
                adjs[vid]=set_tmp
                adjs[vid].add(nid)
                count+=1
            if adjs.__contains__(nid):
                if not adjs[nid].__contains__(vid):
                    adjs[nid].add(vid)
            else:
                set_tmp=set()
                adjs[nid]=set_tmp
                adjs[nid].add(vid)
    print(count)
    # print('aaaaaaaa')
    # start to write
    metisFileWrite=open(edgesMetisFileName,'w+')
    metisFileWrite.write(str(nodeNum)+' '+str(count)+'\n')
    for i in range(nodeNum):
        if adjs.__contains__(i):
            neighbStr=''
            for neibor in adjs[i]:
                neighbStr+=str(neibor+1)+' '
            neighbStr=neighbStr[:-1]
            neighbStr+='\n'
            metisFileWrite.write(neighbStr)
        else:
            neighbStr=''
            neighbStr+='\n'
            metisFileWrite.write(neighbStr)
            print("not contains {0}".format(i))
    metisFileWrite.close()


