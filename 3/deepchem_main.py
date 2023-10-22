import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import deepchem as dc
from fnn import FeedforwardNeuralNetwork


torch.manual_seed(43)

def loss_evaluator(dataloader, model, loss_func):
    sample_size = len(dataloader.dataset)
    pred_list = []
    true_list = []
    with torch.no_grad():
        loss = 0
        for each_X, each_y in dataloader:
            each_pred = model.forward(each_X)
            pred_list.extend(each_pred.tolist())
            true_list.extend(each_y.tolist())
            loss += loss_func(each_pred, each_y).item()
    return loss / sample_size, pred_list, true_list

featurizer = dc.feat.RDKitDescriptors()
tasks, datasets, transformers \
    = dc.molnet.load_bace_regression(featurizer)

train_set, val_set, test_set = datasets
train_dataloader = DataLoader(
    TensorDataset(
        torch.FloatTensor(train_set.X),
        torch.FloatTensor(train_set.y)),
    batch_size=32, shuffle=True)
val_dataloader = DataLoader(
    TensorDataset(torch.FloatTensor(val_set.X),
                  torch.FloatTensor(val_set.y)),
    batch_size=32,
    shuffle=True)
test_dataloader = DataLoader(
    TensorDataset(torch.FloatTensor(test_set.X),
                  torch.FloatTensor(test_set.y)),
    batch_size=32,
    shuffle=True)

model = FeedforwardNeuralNetwork(in_dim=train_set.X.shape[1],
                                 hidden_dim=32)
loss_func = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3,
                             weight_decay=1e-5)
n_step = 0
train_loss_list = []
val_loss_list = []
for each_epoch in range(30):
    for each_X, each_y in train_dataloader:
        if n_step % 10 == 0:
            train_loss, _, _ = loss_evaluator(train_dataloader,
                                              model,
                                              loss_func)
            val_loss, _, _ = loss_evaluator(val_dataloader,
                                            model,
                                            loss_func)
            if n_step % 100 == 0:
                print('step: {},\t\ttrain loss: {}'.format(
                    n_step, train_loss))
                print('step: {},\t\tval loss: {}'.format(
                    n_step, val_loss))
            train_loss_list.append((n_step, train_loss))
            val_loss_list.append((n_step, val_loss))

        each_pred = model.forward(each_X)
        loss = loss_func(each_pred, each_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_step += 1


fig, ax = plt.subplots(1, 1)
ax.plot(*list(zip(*train_loss_list)), marker='+')
ax.plot(*list(zip(*val_loss_list)), marker='.')
ax.set_title('Learning curve')
ax.set_xlabel('# of updates')
ax.set_ylabel('Loss')
ax.set_yscale('log')
plt.savefig('bace_loss.pdf')
plt.clf()

test_loss, pred_list, true_list = loss_evaluator(
    train_dataloader, model, loss_func)
print('test_loss: {}'.format(test_loss))

fig, ax = plt.subplots(1, 1)
ax.scatter(pred_list, true_list)
ax.set_title('Test loss = {}'.format(test_loss))
ax.set_xlabel('Predicted normalized pIC50')
ax.set_ylabel('True normalized pIC50')
plt.savefig('bace_scatter.pdf')
plt.clf()
