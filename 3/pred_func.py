import torch
from torch import nn

torch.manual_seed(46)
m = nn.Linear(in_features=5, out_features=1) 
print('model weight: {}'.format(m.weight))
print('model bias: {}'.format(m.bias))

x = torch.ones(5)
y = m(x)
print('input: {}'.format(x))
print('output: {}'.format(y))

X = torch.arange(10, dtype=torch.float).reshape(2, 5)
Y = m(X)
print('input: {}'.format(X))
print('output: {}'.format(Y))

activation = nn.Sigmoid()
Y_bar = activation(m(X))
print('output: {}'.format(Y_bar))

target1 = torch.tensor([0., 1.])
target2 = torch.tensor([1., 0.])
loss_func = nn.BCELoss()
loss1 = loss_func(Y_bar.reshape(-1), target1)
loss2 = loss_func(Y_bar.reshape(-1), target2)
print('target: {}\tloss: {}'.format(target1, loss1))
print('target: {}\tloss: {}'.format(target2, loss2))
