import numpy as np
from collections import defaultdict
def load_cora():
    num_nodes = 2708  # 节点数
    num_feats = 1433  # 特征数
    feat_data = np.zeros((num_nodes, num_feats))  # 构建特征值矩阵
    labels = np.empty((num_nodes), dtype=np.int64)  # 构建标签矩阵
    node_map = {}  # 节点映射
    label_map = {}  # 标签映射
    featsClass=open("/mnt/data/cora/featsClass_raw.txt",'w+')
    edges=open("/mnt/data/cora/edges_raw.txt",'w+')


    with open("/mnt/data/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            fcStr=str(i)+'\t'+'\t'.join(map(str,info[1:-1]))+'\t'+str(labels[i])+'\n'
            featsClass.write(fcStr)
    featsClass.close()

    adj_lists = defaultdict(set)  # 邻接表
    with open("/mnt/data/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            str1=str(paper1)+'\t'+str(paper2)+'\n'
            str2=str(paper2)+'\t'+str(paper1)+'\n'
            edges.write(str1)
            edges.write(str2)
    # with open(featsClass,'w+') as fw:
    edges.close()
    print("process end")



load_cora()

