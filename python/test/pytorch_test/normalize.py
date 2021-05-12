import torch.nn.functional as F
import torch as torch

a_array=[[1,2,3,4],[5,6,7,8]]
a = torch.tensor(a_array, dtype=torch.float32)

print(a)
a=F.normalize(a, p=2, dim=0)  # 对指定维度进行运算,P是范数

print(a)