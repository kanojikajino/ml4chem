import torch
X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
y = torch.tensor([1., 2.])
w = torch.tensor([1., 1., 1.], requires_grad=True)
loss = 0.5 * torch.sum((y - X @ w) ** 2)
print('loss = {}'.format(loss))
loss.backward()
print('grad = {}'.format(w.grad))
