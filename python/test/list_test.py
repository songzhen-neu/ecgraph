# 会报越界错误，因此需要append
list1=[None]*6
list1[0]=0
print(list1[0])

list1[5]=1
print(list1)