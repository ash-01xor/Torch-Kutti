import numpy as np

class Tensor:

    def __init__(self,data) -> None:
        self.data = data if isinstance(data,np.ndarray) else np.array(data)
        self._grad = None
        self._ctx = None

    def __add__(self,other):
        result = Tensor(self.data + other.data)
        result._ctx = Function(Add, self, other)
        return result

    def __mul__(self,other):
        result = Tensor(self.data * other.data)
        result._ctx = Function(Mul, self, other)
        return result
    
    def __repr__(self) -> str:
        return f"tensor({self.data})"
    

class Function:
    def __init__(self,op,*args):
        self.op = op
        self.args = args

    def backward(self,grad):
        return self.op.backward(self.args,grad)


class Add:
    @staticmethod
    def forward(x,y):
        return Tensor(x.data+y.data)
    
    @staticmethod
    def backward(args,grad):
        return grad, grad
    
class Mul:
    @staticmethod
    def forward(x,y):
        return Tensor(x.data * y.data)
    
    @staticmethod
    def backward(args,grad):
        x,y = args
        return Tensor(y.data * grad.data), Tensor(x.data * grad.data)