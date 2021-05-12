import torch

# don't init
x=torch.empty(5,3)
print(x)

# init randomly
y=torch.rand(5,3)
print(y)

x=torch.zeros(5,3,dtype=torch.long)
print(x)
print(x.type())

x=torch.tensor([5.5,3])
print(x)

# z has the same size
z=torch.ones_like(x,dtype=torch.float)
print(z)

print(x.size())

print(torch.add(x,z))

x=torch.zeros(5,3,dtype=torch.long)
print(x)

y=torch.ones_like(x,dtype=torch.float)
print(y)

result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
print(result.type())

result=x+y
print(result)
print(result.type())

x=torch.zeros(5,3,4)
print(x)

x=torch.zeros(5)
print(x)

y=torch.ones_like(x)
print(y)

print(x[0:3])

# if we want to change the x, we need to use "_"
# x.add_(y)
x.copy_(y)
print(x)

x=torch.randn(4,4)
y=x.view(16)
# inferred from other dims
z=x.view(-1,8)
print(x)
print(y)
print(z)

x=torch.randn(1)
print(x)
print(x.item())

