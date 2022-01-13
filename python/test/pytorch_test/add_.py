import torch
tensor=torch.ones(4,4,dtype=torch.float64)
tensor[:,3]=2
print(tensor,"\n")
# tensor+0.5*tensor
tensor=tensor.add(0.5,tensor)
print(tensor)