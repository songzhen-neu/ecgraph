import torch

x=torch.randn(2,3)
x=torch.cat((x,x,x),dim=0)
print(x)
print(x.size())