import matplotlib.pyplot as plt
import torch
from torch import nn

class FeedforwardNeuralNetwork(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fnn(x)

torch.manual_seed(0)
in_dim = 2
sample_size = 1000
X = torch.randn((sample_size, in_dim))
w = torch.randn(in_dim)
y = torch.sin(X @ w).reshape(-1, 1)
model = FeedforwardNeuralNetwork(in_dim=in_dim, hidden_dim=2)
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
plt.savefig('fnn.pdf')
plt.clf()
