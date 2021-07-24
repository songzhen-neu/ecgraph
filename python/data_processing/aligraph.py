# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Preprocess cora dataset and generate node, edge, train, val, test table.
Used by GCN, GAT, GraphSage supervised training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse as sp
import time

# dataNum = 19717
# featDim = 500
# dataName='pubmed'
# train_num=12816
# val_num=1971
# test_num=4930

dataNum = 2449029
featDim = 100
dataName='ogbn-products'
train_num=196615
val_num=39323
test_num=2213091

# dataNum = 232965
# featDim = 602
# dataName='reddit-small'
# train_num=153932
# val_num=23699
# test_num=55334

# dataNum = 2708
# featDim = 1433
# dataName='cora'
# train_num=1408
# val_num=300
# test_num=1000


abspath=os.getcwd()

def preprocess(dataset):
    global abspath
    # process node table
    # dataset=abspath + '/dataprocess/'+dataset
    dataset='/mnt/data/'+dataset
    node_table = "{}/node_table".format(dataset)
    edge_table = "{}/edge_table".format(dataset)
    edge_table_with_self_loop = '{}/edge_table_with_self_loop'.format(dataset)
    train_table = "{}/train_table".format(dataset)
    val_table = "{}/val_table".format(dataset)
    test_table = "{}/test_table".format(dataset)

    print('idx_features_labels done')
    print(edge_table_with_self_loop)
    # if not os.path.exists(node_table):
    featClassRead = open(dataset + "/featsClass.txt", 'r')
    fw=open(node_table,'w')
    fw.write("id:int64" + "\t" + "label:int64" + "\t" + "feature:string" + "\n")
    allLines = featClassRead.readlines()
    for eachLine in allLines:
        featClassStr = ''
        feat = []
        lineSplit = eachLine.split('\t')
        lineSplit[-1]=lineSplit[-1][:-1]
        for dimId in range(1, featDim + 1):
            feat.append(float(lineSplit[dimId]))
        feat=sp.csc_matrix(feat)
        feat = np.array(feat.todense())

        # feat = feature_normalize(sp.csc_matrix(feat))
        # feat = np.array(feat.todense())
        featClassStr+=lineSplit[0]+'\t'+lineSplit[-1]+'\t'+':'.join(map(str,feat[0]))+'\n'
        fw.write(featClassStr)
    fw.close()

    with open(train_table, 'w') as f:
        f.write("id:int64" + "\t" + "weight:float" + "\n")
        for i in range(train_num):
            f.write(str(i) + "\t" + str(1.0) + "\n")
    with open(val_table, 'w') as f:
        f.write("id:int64" + "\t" + "weight:float" + "\n")
        for i in range(train_num, train_num+val_num):
            f.write(str(i) + "\t" + str(1.0) + "\n")
    with open(test_table, 'w') as f:
        f.write("id:int64" + "\t" + "weight:float" + "\n")
        for i in range(train_num+val_num, train_num+val_num+test_num):
            f.write(str(i) + "\t" + str(1.0) + "\n")
    featClassRead.close()

    # process edge table
    edgesRead = open(dataset + "/edges.txt",'r')
    edgeTableWrite=open(edge_table,'w')
    edgeSelfTableWrite = open(edge_table_with_self_loop, 'w')
    edgeTableWrite.write("src_id: int64" + "\t"
                         + "dst_id: int64" + "\t"
                         + "weight: double" + "\n")

    edgeSelfTableWrite.write("src_id: int64" + "\t"
                             + "dst_id: int64" + "\t"
                             + "weight: double" + "\n")

    allLines=edgesRead.readlines()
    for eachLine in allLines:
        lineSplit = eachLine.split('\t')
        lineSplit[-1]=lineSplit[-1][:-1]
        edge_tmp=lineSplit[0]+'\t'+lineSplit[1]+'\t'+'0.0'+'\n'
        edgeTableWrite.write(edge_tmp)
        if lineSplit[0]!=lineSplit[1]:
            edgeSelfTableWrite.write(edge_tmp)
    for i in range(dataNum):
        edgeSelfTableWrite.write(str(i)+ '\t' + str(i) + '\t' + '0.0' + '\n')

    edgesRead.close()
    edgeTableWrite.close()
    edgeSelfTableWrite.close()

    print("Data Process Done.")
    return
    print("Data {} has exist.".format(dataset))


def encode_label(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: i for i, c in
                    enumerate(classes)}
    labels_int64 = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int64)
    return labels_int64


def feature_normalize(sparse_matrix):
    """Normalize sparse matrix feature by row.
    Reference:
      DGL(https://github.com/dmlc/dgl).
    """
    row_sum = np.array(sparse_matrix.sum(1))
    row_norm = np.power(row_sum, -1).flatten()
    row_norm[np.isinf(row_norm)] = 0.
    row_matrix_norm = sp.diags(row_norm)
    sparse_matrix = row_matrix_norm.dot(sparse_matrix)
    return sparse_matrix


if __name__ == "__main__":
    start=time.time()
    preprocess(dataName)
    end=time.time()
    print("time:{0}".format(end-start))
