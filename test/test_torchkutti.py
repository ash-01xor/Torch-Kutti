import unittest
import sys
sys.path.append(".")

import numpy as np
from Torch_Kutti.tensor_ops import Tensor

class TestTensorOperations(unittest.TestCase):
    def setUp(self):
        self.tensor_a = Tensor([1, 2, 3], requires_grad=True)
        self.tensor_b = Tensor([4, 5, 6], requires_grad=True)

    def test_addition(self):
        result =  self.tensor_b + self.tensor_a 

        
        self.assertIsInstance(result, Tensor)
        self.assertTrue(np.array_equal(result.data(), np.array([5, 7, 9])))

        
        result.backward()
        self.assertTrue(np.array_equal(self.tensor_a.grad, np.array([1, 1, 1])))
        self.assertTrue(np.array_equal(self.tensor_b.grad, np.array([1, 1, 1])))

    def test_multiplication(self):
        prod = self.tensor_a * self.tensor_b
        
        self.assertIsInstance(prod,Tensor)
        self.assertTrue(np.array_equal(prod.data(), np.array([4, 10, 18])))

        prod.backward()
        self.assertTrue(np.array_equal(self.tensor_a.grad, np.array([4, 5, 6])))
        self.assertTrue(np.array_equal(self.tensor_b.grad, np.array([1, 2, 3])))

if __name__ == '__main__':
    unittest.main()
