from torch.autograd import Variable
import torch

mask = Variable(torch.zeros(5, 8))
mask[0:3, [1, 2, 3]] = 1
mask[3:5, [4, 6]] = 1
print(mask)

# sum row
deg = mask.sum(1, keepdim=True)
print(deg)
# sum column
a = mask.sum(0, keepdim=True)
print(a)
# sum all
a = mask.sum()
print(a)

b = mask.div(5)
print(b)

b = mask.div(deg)
print(b)
