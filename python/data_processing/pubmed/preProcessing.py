import numpy as np
from collections import defaultdict
import sys

def load_pubmed():  # pubmed数据集部分和cora的基本相同
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    edgeNum=0

    # print(sys.argv[0])
    with open("../../../data_raw/pubmed/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("../../../data_raw/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            if paper1!=paper2:
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)


    with open('../../../data/pubmed/edges.txt','w+') as fw:
        for i in range(19717):
            for neiId in adj_lists[i]:
                str_tmp=str(i)+'\t'+str(neiId)+'\n'
                edgeNum+=1
                fw.write(str_tmp)
    with open('../../../data/pubmed/featsClass.txt','w+') as fw:
        for i in range(19717):
            str_tmp=str(i)+'\t'
            for j in range(500):
                str_tmp+=str(feat_data[i][j])+'\t'
            str_tmp+=str(labels[i][0])+'\n'
            fw.write(str_tmp)
    print(edgeNum)
    return feat_data, labels, adj_lists


if __name__=='__main__':
    load_pubmed()