import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, operation=None):
        self._data = array(data)
        self.requires_grad = requires_grad
        self.operation = operation
        self.child = []
        self.shape = self._data.shape
        if self.requires_grad:
            self.grad = np.zeros_like(data)

    def __repr__(self):
        return f"({self._data}, requires_grad={self.requires_grad})"

    def data(self):
        return self._data
    
    def tolist(self):
        return self._data.tolist()
    
    def toarray(self):
        return self._data
    
    def zero_grad(self):
        self.grad = np.zeros_like(self._data)

    def zero_grad_tree(self):
        self.zero_grad()
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grad_tree()
            self.operation = None

    
    def backward(self, grad=None, z=None):
        
        if not self.requires_grad:
            return "requires_grad was set to False"
        
        if grad is None:
            grad = np.ones_like(self._data)

        self.grad += grad

        if z is not None:
            self.child.remove(z)
        
        if self.operation:
            if not self.child:
                self.operation.backward(self.grad, self)
    
    def __add__(self, other):
        op = Add()
        return op.forward(self, tensor(other))
    
    def __radd__(self,other):
        op = Add()
        return op.forward(self,tensor(other))
    
    def __iadd__(self,other):
        op = Add()
        return op.forward(self,tensor(other))
    
    def __neg__(self):
        op = Neg()
        return op.forward(self)
    
    def __sub__(self,other):
        return self + -other
    
    def __rsub__(self,other):
        return other + -self
    
    def __isub__(self,other):
        return self + -other
    
    def __matmul__(self,other):
        op = MatMul()
        return op.forward(self,tensor(other))
    
    def __mul__(self,other):
        op = Mul()
        return op.forward(self,tensor(other))
    
    def __imul__(self,other):
        op = Mul()
        return op.forward(self,tensor(other))

    def __rmul__(self,other):
        op = Mul()
        return op.forward(self,tensor(other))
    
    def __truediv__(self,other):
        op = Div()
        return op.forward(self,tensor(other))

    def sum(self, dim=-1, keepdims=False):
        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)
    

class Add:
    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
        data = a._data + b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)
        a.child.append(z)
        b.child.append(z)
        self.cache = (a, b)

        return z
    
    def backward(self, dz, z):
        a, b = self.cache

        if a.requires_grad:
            da = dz

            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)

        for n, dim in enumerate(a.shape):
            if dim == 1:
                da = da.sum(axis=n, keepdims=True)
        a.backward(da, z)

        if b.requires_grad:
            db = dz

            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

        for n, dim in enumerate(a.shape):
            if dim == 1:
                db = db.sum(axis=n, keepdims=True)
        b.backward(db, z)

class Neg:

    def forward(self,a):
        requires_grad = a.requires_grad
        data = - a._data
        z  = Tensor(data,requires_grad=requires_grad,operation=self)
        self.parents = (a,)
        a.child.append(z)
        self.cache = a
        return z
        

    def backward(self,dz,z):
        a = self.cache

        if a.requires_grad:
            da = -dz
            a.backward(da,z)

class MatMul:
    def forward(self,a,b):

        requires_grad = a.requires_grad or b.requires_grad
        data = a._data @ b._data
        
        z = Tensor(data,requires_grad=requires_grad,operation=self)
        
        self.parents = (a,b)
        a.child.append(z)
        b.child.append(z)
        self.cache = (a,b)

        return z
    
    def backward(self,dz,z):

        a,b = self.cache

        if a.requires_grad:

            da = dz @ b._data.swapaxes(-1,-2)

            in_dim = len(a.shape)
            grad_dim = len(da.shape)

            for _ in range(grad_dim-in_dim):
                da = da.sum(axis=0)

            a.backward(da,z)

        if b.requires_grad:

            db = a._data.swapaxes(-1,-2) @ dz

            in_dim = len(b.shape)
            grad_dim = len(db.shape)

            for _ in range(grad_dim-in_dim):
                db = db.sum(axis=0)

            b.backward(db,z)
            
class Mul:

    def forward(self,a,b):
        requires_grad = a.requires_grad or b.requires_grad
        
        data = a._data * b._data

        z = Tensor(data,requires_grad=requires_grad,operation=self)

        self.parents = (a,b)
        a.child.append(z)
        b.child.append(z)
        self.cache = (a,b)

        return z

    def backward(self,dz,z):
        
        a,b = self.cache

        if a.requires_grad:
            da = dz * b._data
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        if b.requires_grad:
            db = dz * a._data
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            
            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)
        
class Div:

    def forward(self,a,b):
        requires_grad = a.requires_grad or b.requires_grad
        data = a._data / b._data
        z = Tensor(data,requires_grad=requires_grad,operation=self)
        self.parents = (a,b)
        a.child.append(z)
        b.child.append(z)

        self.cache = (a,b)
        return z


    def backward(self,dz,z):
        a, b = self.cache

        if a.requires_grad:
            da = dz * (1/b._data)

            grad_dim = len(dz.shape)
            in_dim = len(a.shape)

            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            for n , dim in enumerate(a.shape):
                if dim==1:
                    da = da.sum(axis=n,keepdims=True)

            a.backward(da,z)

        if b.requires_grad:
            db = dz * (1/a._data)

            grad_dim = len(dz.shape)
            in_dim = len(b.shape)

            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            for n , dim in enumerate(b.shape):
                if dim==1:
                    da = db.sum(axis=n,keepdims=True)

            b.backward(db,z)

class Sum:
    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
        data = a._data.sum(axis=dim, keepdims=keepdims)
        z = Tensor(data, requires_grad=requires_grad, operation=self)

        self.parents = (a,)
        a.child.append(z)
        self.cache = (a)
        return z
    
    def backward(self, dz, z):
        a = self.cache

        if a.requires_grad:
            da = np.ones(a.shape) * dz
            a.backward(da, z)


# helper functions
def array(data):
    if isinstance(data, Tensor):
        return data.toarray()
    else:
        return np.array(data)

def tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)

