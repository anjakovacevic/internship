import math
import numpy as np
import matplotlib.pyplot as plt

#from micrograd.engine import Value
from micrograd import nn

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)

# plt.show()

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

'''
h = 0.0001 #slope for norm.
a = 2.0
b = -3.0
c = 10.0
d1 = a*b+c
a+=h
d2 = a*b+c
print('d1',d1)
print('d2',d2)
print('slope', (d2-d1)/h)
# peske izvod za 'a' (otp), ako promenimo b ili c za h, onda po njima radimo izvod
'''
'''
a = Value(2.0)
# print(a)
b = Value(-3.0)
c = Value(10.0)
d = a*b+c
print(d._prev) # getting the children of this value
f = Value(-2.0)
L = d*f
print(L)
'''

'''
# Now, backprop in the respect for L
# In comparison to NNs, a b c are weights and L is loss func
def bp():
    h=0.0001

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a*b
    d = e+c
    f = Value(-2.0)
    L = d*f
    L1 = L.data

    a = Value(2.0)
    a.data+=h
    b = Value(-3.0)
    c = Value(10.0)
    e = a*b
    d = e+c
    f = Value(-2.0)
    L = d*f
    L2 = L.data
    print((L2-L1)/h)

bp()
'''

'''
# Backpropagation on a neuron

# # using sigmoid as an activation function
# def tanh(n):
#     x= n.data
#     t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
#     out = Value(t, (n, ), 'tanh')
#     return out

# inputs
x1 = Value(2.0)
x2 = Value(0.0)
# weights
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432) #bias
# x1*w1 + x2*w2 + b <-- what we are trying to calc
x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh() #output with the activation func
print(o)

# now we need to backprop through output
# derivative of sigmoid is in relation to n is 1-sig**2

o.backward()
print(o)
'''

a = Value(3.0)
b = a+a
b.backward()
#print(b.grad)

#
a = Value(-2.0)
b = Value(3.0)
d=a*b
e=a+b
f=d*e
f.backward()
print(a.grad, b.grad, d.grad, e.grad, f.grad)