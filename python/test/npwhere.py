import torch.nn.functional as F
import torch
import numpy as np
a = np.random.randint(-5, 5, (5, 5))
print(a)

x = F.normalize(torch.FloatTensor(a), p=2, dim=1)

print(np.where(a > 0, a, 0))
print(x)