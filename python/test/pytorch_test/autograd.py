import torch
import torch.nn as nn
import numpy as np

x = torch.ones(2, 2, requires_grad=True)
print(x)
x.data=torch.FloatTensor(np.array([[3,2],[2,2]]))

y=x*x



y_part=torch.FloatTensor(np.array([[1,1]]))

y_cat=torch.cat((y,y_part),0)

z=y_cat.mean()

z.backward()
# print()
print(y_cat.grad)

print(y.grad)

print(x.grad)

y = torch.ones(2, 2, requires_grad=False)

print(y)
w = torch.ones(2, 5, requires_grad=True)
z = (x+y * y * 3).mm(w)

out = z.mean()

print(z)
print(out)

out.backward()

# mean grad: div size
print(w.grad)
print(x.grad)

x = torch.ones(3, requires_grad=True)
y = x * 2

# while y.data_raw.norm() < 1000:
#     # ||y||_2
#     # print(y.data_raw.norm())
#     y = y * 2
# print(y)

z=pow(y,10)
print(z)

z2=z*10

v=torch.tensor([0.1,0.01,0.001],dtype=torch.float)
# equals to y*v
# 从z2开始反向传播, 最后会对应每一维乘以矩阵v, 如果写z.backward, 则从z开始反向传播
z2.backward(v)
print(x.grad)


# 可以将代码包裹在with torch.no_grad()中，停止对.requires_grad=True的张量自动求导
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)