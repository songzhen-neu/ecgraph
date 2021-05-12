import numpy as np
import scipy.sparse as sp


adj=sp.coo_matrix((np.ones(3),([0,0,2],[0,1,1])),shape=(3,3),dtype=np.int)
print(adj)

print("*********************")

print(adj.T)

print("*********************")

print(adj.T.multiply(adj.T > adj))

print("**********adj.T > adj***********")
print((adj.T>adj))

print("**********multiply(2)***********")
print(adj.T.multiply(2))

print("*********adj+adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)************")

print(adj+adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))


print("**********adj+adj.T.multiply(adj.T > adj) ***********")

print(adj+adj.T.multiply(adj.T > adj))



# print("++++++++++++++++")
# b=np.array([[1,0],[1,0]])
# print(b)
# print("*********************")
# print(b.T)
# print("*********************")
# print(b.T.multiply(b.T>b))


