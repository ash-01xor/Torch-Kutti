import numpy as np
from torch_kutti import Tensor

x = Tensor([6])
y = Tensor([9])    
z = x*y
print(z)
z.backward()
print(f"x:{x.data},grad{x._grad.data}")
print(f"y:{y.data} grad:{y._grad.data}")