
import numpy as np
from cmake.build.example3 import *
import time
# print(example.add(1,2))
#
# var1=example.add_arrays_1d(np.array([1,3,5,7,9]),
#                             np.array([2,4,6,8,10]))
# print(var1)
#
#
# var2 = example.add_arrays_2d(np.array(range(0,16)).reshape([4, 4]),
#                               np.array(range(20,36)).reshape([4, 4]))
# print('-'*50)
# print('var2', var2)


# A = [[1,2,3,4],[5,6]]
#
# B = example.modify2()
#
# print(B)
#
# C=example.test3()
# print(C)

# d=Dog()
# d.name='Tom'
# print(d.name)
#
# d.food=['a','b']
#
# print(d.food)
#
# d.noname={
#     1:[1,2],
#     2:[3,4]
# }
#
# print(d.noname)

input1=np.ones(shape=(2708,1433),dtype=float)
input2=np.ones(shape=(2708,1433),dtype=float)

start=time.time()
add_arrays_2d(input1,input2)
end=time.time()
print("add time:{0}".format(end-start))

start=time.time()
dict_1=dict()
for i in range(2708):
    list_1=list()
    for j in range(1000):
        # a=1
        list_1.append(j)
    dict_1[i]=list_1

end=time.time()
print("for time:{0}".format(end-start))

start=time.time()
sendMatrix(dict_1,dict_1)
end=time.time()
print("sendMatrix time:{0}".format(end-start))
print('aa')

start=time.time()
aaa=receiveMatrix()
end=time.time()
print("receiveMatrix time:{0}".format(end-start))