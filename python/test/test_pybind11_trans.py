from cmake.build.example2 import *
import numpy as np
import time


a=[]
for i in range(6):
    b=[]
    for j in range(200000):
        b.append(j)
    a.append(b)
# a=np.array(a)
# a=np.ndarray([2708,1433])

start=time.time()
test_list_list(a)
end=time.time()

print("time_list:{:.4f}".format(end-start))

# start=time.time()
# test_list2(a)
# end=time.time()
#
# print("time_list:{:.4f}".format(end-start))

# start=time.time()
# test_list3(a)
# end=time.time()
#
# print("time_list:{:.4f}".format(end-start))
#
# b={}
#
# for i in range(2000):
#     b[i]=i
#
# start=time.time()
# test_dict(b)
# end=time.time()
#
# print("time_dict:{:.4f}".format(end-start))


# start=time.time()
# for i in b:
#     # print(str(i)+","+str(b[i]))
#     # c=1
#     pass
# end=time.time()
#
# print("time_dict:{:.4f}".format(end-start))