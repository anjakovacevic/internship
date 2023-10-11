# Implementing NN using MLP and a research paper Bengio et al 2003
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
# print(words[:8])
# print(len(words)) # 32033

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
# print(itos)  {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e',...

# build the dataset
block_size = 3 # how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:5]:
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
  
X = torch.tensor(X) # tensor([[0,  0,  0],[0,  0,  5],[0,  5, 13],[26, 26, 25],[26, 25, 26],[25, 26, 24]])
Y = torch.tensor(Y) # tensor([ 5, 13, 13,  ..., 26, 24,  0])
#print(X, Y)
# print(X.shape, Y.shape, X.dtype, Y.dtype) 
# torch.Size([228146, 3]) torch.Size([228146]) torch.int64 torch.int64
 
# In the paper, 17k words were put in a 30 dim space
# Now, we are going to put our 27 char into 2 dims

# implementing the embedding lookup table
C = torch.randn((27, 2))
# 1st representation:
# F.one_hot(torch.tensor(5), num_classes=27) #must be tensor, not an int
# 5th dimention is 1, the rest are 0
# 2nd representation:
# C[5]
# C[torch.tensor([5, 6, 7])]
# C[X][13, 2]
emb = C[X] # shape: ([32, 3, 2]) we take 3 and 2, 2dim embeddings and there is 3 of them
W1 = torch.randn((6, 100)) # num of inputs is 6 and number of neurons is 100
b1 = torch.randn(100)
# we want to do: emb @ W1 + b1, but dims don't agree. [32, 3, 2] --> [32, 6]
# emb[:, 0, :] # [32, 2]
# cat = concatinate, 0 1 2 is for block size 3
# torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :], 1])
# 2nd option is unbind, which removes one dimension
# torch. cat(torch.unbind(emb, 1), 1)
# 3rd option
# emb.view(32, 6) == torch.cat(torch.unbind(emb, 1),1)
# radi dobro jer ne menja stanje memorije, samo prikaz za manipulacije
# ===>
h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
# print(h.shape)  #torch.Size([228146, 100])
# concat is way less efficient because it creates whole new memory 

# creating the final layer
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
logits = h @ W2 + b2 # shape [32, 27]

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True) # shape [32, 27]
# negative log likelihood loss = 16.7732 which we want to minimize
loss=-prob[torch.arange(32), Y].log().mean()
print(loss)

