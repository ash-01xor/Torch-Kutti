import numpy as np
from torch_kutti import Tensor

def f(x):
    return x * x +  x

x = Tensor([6])
y = Tensor([9])    
z = x+y
print(z)
z.backward()
print(f"x:{x},grad:{x.grad}")
print(f"y:{y} grad:{y.grad}")
print("--"*25)

x1 = Tensor([2])
y1 = Tensor([3]) 
z1 = x1*y1
print(z1)
z1.backward()
print(f"x:{x1},grad:{x1.grad}")
print(f"y:{y1} grad:{y1.grad}")

print("--"*25)

x2 = Tensor([3.2])
z2 = f(x2)
z2.backward()
print(f"X:{x2} grad:{x2.grad}")

