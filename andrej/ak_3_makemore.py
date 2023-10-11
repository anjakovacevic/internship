words = open('names.txt', 'r').read().splitlines()
# ispitivanje dataseta
'''
print(words[:10])
print("kolicina reci:",len(words),
    "\nminimalna duzina reci:",min(len(w) for w in words),
    "\nmaximalna duzina reci:",max(len(w) for w in words))
'''

'''
# BIGRAM LANGUAGE MODEL (always working w 2 char and focusing on what comes after one another)
#tuple of 2 char in a dictionary  
b = {}
for w in words:
    # special start token START END
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # if its not in the bigraam return 0
# print(b)
# now we are going to count how many times does an iteration of 2 char happen
# sort how many times does a combination occur

print(sorted(b.items(), key = lambda kv:-kv[1]))
'''

import torch

# 28x28 array for 26 letter of alphabet and 2 special char
N = torch.zeros((27, 27), dtype=torch.int32)

# lookup table, trowing out duplicates, maping the char to int
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1

# print(N)
import matplotlib.pyplot as plt

'''
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
'''

'''
# create the probability vector that 
p = N[0].float()
p = p / p.sum()
# float is so we could normalize it
#print(p)

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#print(itos[ix]) so we can see the char
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
# print(p)

# normalization of the row of probabilities
P=(N+5).float()
P /= P.sum(1, keepdim=True)
# KORISTITI OVAKVU ANOTACIJU UMESTO P/P BRZE IZVRSAVANJE
# KEEPDIM OBAVEZNO!!!
#print(P.sum(1))
#print(P.sum(1, keepdim=True).shape)
# 27, 27
# 27, 1    :) broadcastable!
'''
'''

g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out = []
    ix = 0
    while True:
        #p = N[ix].float()
        #p = p/p.sum()
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  #That is now the end token
            break
    print(''.join(out))
'''

'''

# look at the bigrams of the first 2 names, emma and olivia
# also look at their probabilities
# lookup logaritmic probability
# likelyhood - a*b*c; log(likelyhood) is log(a)+log(b)+log(c)

# GOAL : maximize likelihood of the data probabilities
# eq to maximizing log likelihood (because log is monotonic function! only grows)
# eq to minimizing the negative log likelihood
# eq to minimizing the avg negative log likelihood

log_likelihood = 0.0
n=0
#for w in words[:3]:
#for w in words:
for w in ["anya"]:
  #turns out anja is unlikely to be a name, to fix this we do smoothing = adding 1 or some other number do P
  # just so its impossible to get -inf on a name
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n+=1
    print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
# log_likelihood goes from 0 to -inf, 0 is best, when all p are 1
nll=-log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

'''

'''
# OVAJ JE SKLEPAN, DOLE JE LEPO NAPISANO
#create the training set of bigrams (x,y)
xs, ys = [], []  # lists

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    #print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
    
xs = torch.tensor(xs)
ys = torch.tensor(ys)
#print(xs)
#print(ys)


import torch.nn.functional as F
#xenc = x encoded
# shows which bit is turned on for compatibility
xenc = F.one_hot(xs, num_classes=27).float()
#print(xenc)
# plt.imshow(xenc)

# normaliyation around 0 
# matrix multiplication operator in pytorch is @ symbol
# 1 here means theres only 1 neuron
# W = torch.randn((27, 1))
# print(xenc@W)

#RAND INIT 27 NEURONS WEIGHTS, EACH NEURON RECIEVES 27 INPUTS
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# FORWARD PASS

#print(xenc@W)   # shape 5x27
# this is telling us what the firing rate on activation on all those neurons is
# for example, (xenc@W)[3, 13] is telling us how the 13th neuron is looking at one of those inputs

# this is also giving us log counts, so we need to exp it to get the same results as is the matplotlib table above
#print(xenc@W.exp())
logits = xenc@W.exp()   #log-count interpretation
counts = logits.exp()   #eq N
probs = counts / counts.sum(1, keepdims=True)  # normalize to get the normal distribution
# these last 2 rows are called softmax activation function!
# print(prob)
# now for every one of our examples we have a row in a NN
# first row tells us how likely is each character to come after a 0 or a "." in this dataset (".emma.")

# how to we measure the propabilities -> with loss function

loss=-probs[torch.arange(5), ys].log().mean()
print("prvi loss",loss.item())

# BACKWARD PASS
W.grad = None       #same as 0
loss.backward()
# print(W.grad)
W.data+= -0.1*W.grad

logits = xenc@W.exp()   
counts = logits.exp()   
probs = counts / counts.sum(1, keepdims=True) 
loss=-probs[torch.arange(5), ys].log().mean()
print("drugi loss",loss.item())
'''
import torch.nn.functional as F


###########
# written pretty i guess?

# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(100):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.1*(W**2).mean()
  #print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad
print(loss.item())

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)
for i in range(9):
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

plt.show()