import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np




torch.manual_seed(1)


x=torch.tensor([[-0.1,-0.2,0.2]],dtype=torch.float64)
x_cat=torch.tensor([[-0.2,-0.5,0.5]],dtype=torch.float64)





x=torch.reshape(x,(1,3))
x_cat=torch.reshape(x_cat,(1,3))


print('x:{0}'.format(x))

w1=torch.rand(3,3,dtype=torch.float64)
print('w1:{0}'.format(w1.data))

w2=torch.rand(6,3,dtype=torch.float64)
print('w2:{0}'.format(w2.data))

# nn.init.normal_(w1)
w1.requires_grad=True
w1.retain_grad()
w2.requires_grad=True
w2.retain_grad()

# m=x.mm(w1)

# w2=torch.tensor(np.ones(shape=(2,2)))
# nn.init.normal_(w2)
# w2.requires_grad=True
# w2.retain_grad()
# y=m.mm(w2)

h1=x.mm(w1)
print("h1:{0}".format(h1.data))
h1.retain_grad()


htmp=torch.cat((h1,x_cat),1)
print('htmp.cat:{0}'.format(htmp.data))

# htmp.requires_grad=True
htmp.retain_grad()

y=htmp.mm(w2)




# print("y=x*w:{0}={1}*{2}".format(y.data_raw,x.data_raw,w1.data_raw))

y.retain_grad()


print('softmax y:{0}'.format(F.softmax(y,dim=1)))
labels=torch.tensor([0],dtype=torch.long)


# softmax是概率，log相当于做了交叉熵，但所有维度都当做标签计算了
# 如，输出维度7维， 每个维度都做了1*logP
z=F.log_softmax(y,dim=1)

# z.requires_grad=True
z.retain_grad()

# 取标签所对应位置的相反数
loss=F.nll_loss(z,labels)


print("z:{0}".format(z))


loss.backward()


print("w1.grad:{0}".format(w1.grad.data))
# print("w2.grad:{0}".format(w2.grad.data_raw))



print("y.grad:{0}".format(y.grad.data))


print("z.grad:{0}".format(z.grad.data))

print("htmp.grad:{0}".format(htmp.grad.data))

print("h1.grad:{0}".format(h1.grad.data))