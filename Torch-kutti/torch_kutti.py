import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = None
        self._ctx = None

    def __add__(self, other):
        """
        Used for addition operation
        Sets the context needed for gradient calculation during backprop 
        """
        result = Tensor(self.data + other.data)
        result._ctx = Function(Add, self, other)
        return result

    def __mul__(self, other):
        """
        Used for multiplication operation
        """
        result = Tensor(self.data * other.data)
        result._ctx = Function(Mul, self, other)
        return result
    
    def backward(self, grad=None):
        """
        Computes the gradients using backprop
        """
        if self._ctx is None: #checks for context
            return 
        
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        
        op = self._ctx.op #retrives the operation performed
        child_nodes = self._ctx.args # used to find the input nodes
        
        grads = op.backward(self._ctx,grad) #computes the gradients
        
        for tensor,grad in zip(child_nodes,grads):
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad
            tensor.backward(grad)

    def __repr__(self):
        return f"tensor({self.data})"

class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

class Add:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(ctx, grad):
        x,y = ctx.args
        return Tensor([1])*grad,Tensor([1])*grad

class Mul:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data)*grad, Tensor(x.data)*grad