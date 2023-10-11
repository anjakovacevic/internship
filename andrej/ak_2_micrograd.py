import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random

'''
# double so it would be float64
# we have to say that trey require gradients
x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

# data and grad attributes
# "o.item()" and "o.data.item()" will produce the same result in pytorch
print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
'''

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
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
  
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():
      self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
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




class Neuron:
  
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]

# list of neurons
class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  # in pytorch parameters returns the tensor, here is similar but its scalar
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
    # same as:
    '''
    params = []
    for neuron in self.neurons:
        ps = neuron.parameters()
        params.extend(ps)
    return params
    '''

# mlp - multilayer of perceptrons
class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

n = MLP(3, [4, 4, 1])
#dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
#ypred = [n(i) for i in xs]
#print(ypred)

'''
# MNE for loss function
# the more we missed, the loss willl be a bigger number. we want the loss to be low
#print("\n",[(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print(loss)
loss.backward()

# ovaj print je negativan sto znaci da ovaj neuron ima negativan uticaj na loss
# sto je dobro za nas i zelimo da povecavamo njegov weight
#print(n.layers[0].neurons[0].w[0].grad)
#print(n.layers[0].neurons[0].w[0].data)

# modifying p.data like gradient descent in the directon of loss
# we use the '-' for minimization
for p in n.parameters():
    p.data += -0.01 * p.grad

ypred = [n(i) for i in xs]
print("print new loss\n",ypred)
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print(loss)
#now we can repeat this process to minimize the loss, with the backward and the grad

loss.backward()
for p in n.parameters():
    p.data += -0.01 * p.grad    #0.01 je brzina promene ka gradijentu - learning rate
    
ypred = [n(i) for i in xs]
print("print new loss\n",ypred)
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print(loss)
'''

# let automate this for the same dataset
for k in range(40):
  
  # forward pass - loss evaluation
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0            ##### reset the gradient !!!
  loss.backward()
  
  # update - simple stochastic gd update
  for p in n.parameters():
    p.data += -0.01 * p.grad
  
  print(k, loss.data)
  