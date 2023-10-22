import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class FeedforwardNeuralNetwork(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.fnn(x)

torch.manual_seed(0)
sample_size = 1000
in_dim = 2
X = torch.randn((sample_size, in_dim))
w = torch.randn(in_dim)
y = torch.sin(X @ w).reshape(-1, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FeedforwardNeuralNetwork(in_dim=in_dim, hidden_dim=10)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
n_step = 0
loss_list = []
for each_epoch in range(100):
    for each_X, each_y in dataloader:
        each_pred = model.forward(each_X)
        loss = loss_func(each_pred, each_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_step += 1
        if n_step % 100 == 0:
            print('step: {},\t\tloss: {}'.format(n_step, loss.item()))
            loss_list.append((n_step, loss.item()))

y_pred = model.forward(X)
vmin = min(y.min(), y_pred.min())
vmax = max(y.max(), y_pred.max())
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 6))
im = ax1.scatter(X[:, 0], X[:, 1], c=y,
                 cmap='gray', vmin=vmin, vmax=vmax)
ax1.set_title('Ground truth')
im = ax2.scatter(X[:, 0], X[:, 1], c=y_pred.detach().numpy(),
                 cmap='gray', vmin=vmin, vmax=vmax)
ax2.set_title('Prediction')
fig.colorbar(im, ax=ax2)
plt.savefig('sgd.pdf')
plt.clf()

fig, ax = plt.subplots(1, 1)
ax.plot(*list(zip(*loss_list)))
ax.set_title('Loss')
ax.set_xlabel('# of updates')
ax.set_ylabel('Loss')
plt.savefig('sgd_loss.pdf')
plt.clf()
