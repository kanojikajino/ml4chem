import torch
from rdkit import Chem
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import OneHotCategorical
from tqdm import tqdm
from smiles_vocab import SmilesVocabulary
from torchdrug.data.molecule import PackedMolecule
from torchdrug.metrics import penalized_logP

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def filter_valid(smiles_list):
    success_list = []
    fail_idx_list = []
    for each_idx, each_smiles in enumerate(smiles_list):
        try:
            smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(each_smiles))
            success_list.append(smiles)
        except:
            fail_idx_list.append(each_idx)
    return success_list, fail_idx_list


class SmilesLSTM(nn.Module):

    def __init__(self, vocab, hidden_size, n_layers):
        super().__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab.char_list)
        self.lstm = nn.LSTM(
            vocab_size,
            hidden_size,
            n_layers,
            batch_first=True)
        self.out_linear = nn.Linear(hidden_size, vocab_size)
        self.out_activation = nn.Softmax(2)
        self.out_dist_cls = OneHotCategorical
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, in_seq):
        in_seq_one_hot = nn.functional.one_hot(
            in_seq,
            num_classes=self.lstm.input_size).to(torch.float)
        out, _ = self.lstm(in_seq_one_hot)
        return self.out_linear(out)

    def loss(self, in_seq, out_seq):
        return self.loss_func(
            self.forward(in_seq).transpose(1, 2),
            out_seq)

    def generate(self, sample_size=1, max_len=100, smiles=True):
        device = next(self.parameters()).device
        with torch.no_grad():
            self.eval()
            in_seq_one_hot = nn.functional.one_hot(
                tensor([[self.vocab.sos_idx]] * sample_size),
                num_classes=self.lstm.input_size).to(
                    torch.float).to(device)
            h = torch.zeros(
                self.lstm.num_layers,
                sample_size,
                self.lstm.hidden_size).to(device)
            c = torch.zeros(
                self.lstm.num_layers,
                sample_size,
                self.lstm.hidden_size).to(device)
            out_seq_one_hot = in_seq_one_hot.clone()
            out = in_seq_one_hot
            for _ in range(max_len):
                out, (h, c) = self.lstm(out, (h, c))
                out = self.out_activation(self.out_linear(out))
                out = self.out_dist_cls(probs=out).sample()
                out_seq_one_hot = torch.cat(
                    (out_seq_one_hot, out), dim=1)
            self.train()
            if smiles:
                return [self.vocab.seq2smiles(each_onehot)
                        for each_onehot
                        in torch.argmax(out_seq_one_hot, dim=2)]
            return out_seq_one_hot


def trainer(
        model,
        train_tensor,
        val_tensor,
        smiles_vocab,
        lr,
        n_epoch,
        batch_size,
        print_freq,
        device):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(train_tensor[:, :-1],
                                  train_tensor[:, 1:])
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = TensorDataset(val_tensor[:, :-1],
                                val_tensor[:, 1:])
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    train_loss_list = []
    val_loss_list = []
    running_loss = 0
    running_sample_size = 0
    batch_idx = 0
    for each_epoch in range(n_epoch):
        for each_train_batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            each_loss = model.loss(each_train_batch[0].to(device),
                                   each_train_batch[1].to(device))
            each_loss = each_loss.mean()
            running_loss += each_loss.item()
            running_sample_size += len(each_train_batch[0])
            each_loss.backward()
            optimizer.step()
            if (batch_idx+1) % print_freq == 0:
                train_loss_list.append(
                    (batch_idx+1,
                     running_loss/running_sample_size))
                print('#update: {},\tper-example '
                      'train loss:\t{}'.format(
                          batch_idx+1,
                          running_loss/running_sample_size))
                running_loss = 0
                running_sample_size = 0
                if (batch_idx+1) % (print_freq*10) == 0:
                    val_loss = 0
                    with torch.no_grad():
                        for each_val_batch in val_data_loader:
                            each_val_loss = model.loss(
                                each_val_batch[0].to(device),
                                each_val_batch[1].to(device))
                            each_val_loss = each_val_loss.mean()
                            val_loss += each_val_loss.item()
                    val_loss_list.append((
                        batch_idx+1,
                        val_loss/len(val_dataset)))
                    print('#update: {},\tper-example '
                          'val loss:\t{}'.format(
                              batch_idx+1,
                              val_loss/len(val_dataset)))
            batch_idx += 1
    return model, train_loss_list, val_loss_list

def rl_trainer(
        model,
        train_tensor,
        train_tgt,
        smiles_vocab,
        n_epoch=1000,
        sample_size=1000,
        batch_size=128,
        print_freq=100,
        device='cuda'):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_list = []
    avg_reward_list = []
    running_loss = 0
    running_sample_size = 0
    batch_idx = 0
    for each_epoch in range(n_epoch):
        rl_tensor = model.generate(sample_size=sample_size,
                                   smiles=False)
        rl_tensor = torch.argmax(rl_tensor, dim=2)
        rl_smiles_list, fail_idx_list = filter_valid(
            [model.vocab.seq2smiles(each_idx_seq)
             for each_idx_seq in rl_tensor])
        if not rl_smiles_list:
            rl_smiles_list = train_tensor[:sample_size]
            plogp_tensor = train_tgt[:sample_size]
        else:
            rl_packed_dataset \
                = PackedMolecule.from_smiles(rl_smiles_list)
            _plogp_tensor = penalized_logP(rl_packed_dataset)
            plogp_tensor = torch.zeros(len(rl_tensor),
                                       dtype=torch.float)
            each_other_idx = 0
            for each_idx in range(len(plogp_tensor)):
                if each_idx in fail_idx_list:
                    plogp_tensor[each_idx] = -30.0
                else:
                    plogp_tensor[each_idx] \
                        = _plogp_tensor[each_other_idx]
                    each_other_idx += 1
            print(' * mean plogp: {}'.format(plogp_tensor.mean()))
            avg_reward_list.append((each_epoch,
                                    plogp_tensor.mean().item()))
        rl_dataset = TensorDataset(rl_tensor[:, :-1],
                                   rl_tensor[:, 1:],
                                   plogp_tensor)
        rl_data_loader = DataLoader(rl_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
        for each_train_batch in tqdm(rl_data_loader):
            optimizer.zero_grad()
            each_reward = each_train_batch[2].to(device)
            each_loss = model.loss(each_train_batch[0].to(device),
                                   each_train_batch[1].to(device))
            each_loss = (each_reward @ each_loss).mean() \
                / len(each_reward)
            running_loss += each_loss.item()
            running_sample_size += len(each_train_batch[0])
            each_loss.backward()
            optimizer.step()
            if (batch_idx+1) % print_freq == 0:
                train_loss_list.append(
                    (batch_idx+1,
                     running_loss/running_sample_size))
                print('#update: {},\tper-example '
                      'train loss:\t{}'.format(
                          batch_idx+1,
                          running_loss/running_sample_size))
                running_loss = 0
                running_sample_size = 0
            batch_idx += 1
    return model, train_loss_list, avg_reward_list
