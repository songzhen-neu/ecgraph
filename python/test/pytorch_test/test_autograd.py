import torch


x = torch.ones(2, 2, requires_grad=False)
print('x:{0}'.format(x))
print('x_requires_grad:{0}'.format(x.requires_grad))


y = x + 2
print(y)
print('y:{0}'.format(y))
# 保留非叶子节点梯度
# y.retain_grad()
# print('y.requires_grad:',y.requires_grad)
print('y_requires_grad:{0}'.format(y.requires_grad))



z = y * y * 3
print('z:{0}'.format(z))
z.requires_grad=True
print('z_requires_grad:{0}'.format(z.requires_grad))
z.retain_grad()

out = z.mean()
print('out:{0}'.format(out))
print('out.grad_fn:{0}'.format(out.grad_fn))

# y.backward()

# out.backward()
#
# print(x.grad)


out.backward()

print('z_grad:{0}'.format(z.grad))
print('y_grad:{0}'.format(y.grad))
print('x_grad:{0}'.format(x.grad))


z.requires_grad=False
x.requires_grad=True
y.requires_grad=True

y.backward()
print('x.grad:{0}'.format(x.grad))