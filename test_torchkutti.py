import numpy as np
from torch_kutti import Tensor

x = Tensor([6])
y = Tensor([9])    
z = x+y
print(z)
z.backward()
print(f"x:{x.data},grad:{x.grad}")
print(f"y:{y.data} grad:{y.grad}")
print("--"*25)

z1 = x*y
print(z1)
z1.backward()
print(f"x:{x.data},grad{x.grad}")
print(f"y:{y.data} grad:{y.grad}")
