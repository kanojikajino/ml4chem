import torch
from torch import nn
torch.manual_seed(43)
dropout_layer = nn.Dropout(p=0.5)
input_tensor = torch.ones(10)
print(' * train mode')
dropout_layer.train()
for _ in range(3):
    print(dropout_layer(input_tensor))
print(' * evaluation mode')
dropout_layer.eval()
for _ in range(3):
    print(dropout_layer(input_tensor))
