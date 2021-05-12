import torch.nn as nn
import torch
from torch.autograd import Variable

embeds = nn.Embedding(10, 2)

input = torch.arange(0, 6).view(3, 2)
print(input)

input=Variable(input)
output=embeds(input)

print(output.size())
print(embeds.weight.size())
print(output)