import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self._grad = None
        self._ctx = None

    def __add__(self, other):
        result = Tensor(self.data + other.data)
        result._ctx = Function(Add, self, other)
        return result

    def __mul__(self, other):
        result = Tensor(self.data * other.data)
        result._ctx = Function(Mul, self, other)
        return result
    
    def backward(self, grad=None):
        if grad is None:
            grad = Tensor([1.])
            self.grad = grad
        
        op = self._ctx.op
        child_nodes = self._ctx.args
        
        grads = op.backward(self._ctx,grad)
        
        for tensor,grad in zip(child_nodes,grads):
            tensor.grad = grad
    
    def __repr__(self):
        return f"tensor({self.data})"

class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def backward(self, ctx, grad):
        return self.op.backward(ctx, grad)

class Add:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(ctx, grad):
        x,y = ctx.args
        return Tensor([1]),Tensor([1])

class Mul:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data), Tensor(x.data)

