import torch.nn as nn
import numpy as np
import time as time
import scipy.sparse as sp
import random
import torch as torch


# define a vector with 1000 dims
# start_time=time.time()
# vec = np.random.rand(1000,100000)
# end_time=time.time()
# print(end_time-start_time)
# print(vec)
# define a weight matrix


# weight = np.random.rand(100000, 100000)


# start_time=time.time()
# result = np.dot(vec, weight)
# end_time=time.time()
# print(end_time-start_time)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float64)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# print(result)

# 将邻接矩阵处理成coo的格式
# edges=[[0,5],[0,1],[0,3],[2,5],[5,2]]
# edges=np.array(edges)
# adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                      shape=(6, 6),
#                      dtype=np.int)
#
# print(adjs)
#
# print("**********")
# adjs = adjs + adjs.T.multiply(adjs.T > adjs) - adjs.multiply(adjs.T > adjs)
# print(adjs)

# vec = np.array([1, 2, 3])
# weight = np.array([[1, 1], [1, 1], [1, 1]])
# result = np.dot(vec, weight)
# print(result)
edges=[]


for i in range(20000):
    for j in range(500):
      edges.append([i,random.randint(0,19999)])


# edges=[[0,5],[0,1],[0,3],[2,5],[5,2]]
edges=np.array(edges)
adjs = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(20000,20000),
                     dtype=np.int)


adjs=sparse_mx_to_torch_sparse_tensor(adjs)

x = np.random.randn(20000, 600)

weight2=np.random.randn(600, 16)

x=torch.tensor(x)
weight2=torch.tensor(weight2)
start=time.time()

#20000*20000 20000*1000
result = torch.spmm(adjs,x)

end=time.time()

print(end-start)

start=time.time()

# 20000*1000    1000*1000
result2 = torch.mm(x,weight2)

end=time.time()

print(end-start)