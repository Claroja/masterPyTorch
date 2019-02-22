import numpy as np
import torch
from torch.autograd import Variable

x = torch.Tensor(2, 2, 2)
print(x)

y = x.view(1, -1)
print(y)

z = x.view(-1, 4)  # the size -1 is inferred from other dimensions
print(z)

t = x.view(8)
