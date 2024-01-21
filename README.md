# Torch-Kutti
- Simple and basic , tiny autograd engine backed by numpy.
- More of an research and educational framework which is similar to Pytorch.

### What to expect
- Nothing, Hopefully be updating this constantly .
- Modules , losses will be implemented.

### What can be done ?
#### Automatic differentiation
```python
import Torch_Kutti as tk 


a = tk.Tensor([1, 2, 3], requires_grad=True)
b = tk.Tensor([4, 5, 6], requires_grad=True)

def f(x,y):
    return (x+y) + x*2 + (x*y*3)

### calculate complex functions
z = f(a,b)
print(z)

#compute gradients
z.backward()

#get gradients of specific tensors
print(a.grad)
print(b.grad)
```


### License
[MIT](./LICENSE)
