import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import OneHotCategorical, Categorical
from tqdm import tqdm
from smiles_vocab import SmilesVocabulary


class SmilesVAE(nn.Module):

    def __init__(
            self,
            vocab,
            latent_dim,
            emb_dim=128,
            max_len=100,
            encoder_params={'hidden_size': 128,
                            'num_layers': 1,
                            'dropout': 0.},
            decoder_params={'hidden_size': 128,
                            'num_layers': 1,
                            'dropout': 0.},
            encoder2out_params={'out_dim_list': [128, 128]}):
        super().__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab.char_list)
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.beta = 1.0

        self.embedding = nn.Embedding(vocab_size,
                                      emb_dim,
                                      padding_idx=vocab.pad_idx)
        self.encoder = nn.LSTM(emb_dim,
                               batch_first=True,
                               **encoder_params)
        self.encoder2out = nn.Sequential()
        in_dim = encoder_params['hidden_size'] * 2 \
            if encoder_params.get('bidirectional', False)\
            else encoder_params['hidden_size']
        for each_out_dim in encoder2out_params['out_dim_list']:
            self.encoder2out.append(
                nn.Linear(in_dim, each_out_dim))
            self.encoder2out.append(nn.Sigmoid())
            in_dim = each_out_dim
        self.encoder_out2mu = nn.Linear(in_dim, latent_dim)
        self.encoder_out2logvar = nn.Linear(in_dim, latent_dim)

        self.latent2dech = nn.Linear(
            latent_dim,
            decoder_params['hidden_size'] \
            * decoder_params['num_layers'])
        self.latent2decc = nn.Linear(
            latent_dim,
            decoder_params['hidden_size'] \
            * decoder_params['num_layers'])
        self.latent2emb = nn.Linear(latent_dim, emb_dim)
        self.decoder = nn.LSTM(emb_dim,
                               batch_first=True,
                               bidirectional=False,
                               **decoder_params)
        self.decoder2vocab = nn.Linear(
            decoder_params['hidden_size'],
            vocab_size)
        self.out_dist_cls = Categorical
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, in_seq):
        in_seq_emb = self.embedding(in_seq)
        out_seq, (h, c) = self.encoder(in_seq_emb)
        last_out = out_seq[:, -1, :]
        out = self.encoder2out(last_out)
        return (self.encoder_out2mu(out),
                self.encoder_out2logvar(out))

    def reparam(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if deterministic:
            return mu
        else:
            return mu + std * eps

    def decode(self, z, out_seq=None, deterministic=False):
        batch_size = z.shape[0]
        h_unstructured = self.latent2dech(z)
        c_unstructured = self.latent2decc(z)
        h = torch.stack([
            h_unstructured[
                :,
                each_idx:each_idx+self.decoder.hidden_size]
            for each_idx in range(0,
                                  h_unstructured.shape[1],
                                  self.decoder.hidden_size)])
        c = torch.stack([
            c_unstructured[
                :,
                each_idx:each_idx+self.decoder.hidden_size]
            for each_idx in range(0,
                                  c_unstructured.shape[1],
                                  self.decoder.hidden_size)])
        if out_seq is None:
            with torch.no_grad():
                in_seq = torch.tensor(
                    [[self.vocab.sos_idx]] * batch_size,
                    device=self.device)
                out_logit_list = []
                for each_idx in range(self.max_len):
                    in_seq_emb = self.embedding(in_seq)
                    out_seq, (h, c) = self.decoder(
                        in_seq_emb[:, -1:, :],
                        (h, c))
                    out_logit = self.decoder2vocab(out_seq)
                    out_logit_list.append(out_logit)
                    if deterministic:
                        out_idx = torch.argmax(out_logit, dim=2)
                    else:
                        out_prob = nn.functional.softmax(
                            out_logit, dim=2)
                        out_idx = self.out_dist_cls(
                            probs=out_prob).sample()
                    in_seq = torch.cat((in_seq, out_idx), dim=1)
                return torch.cat(out_logit_list, dim=1), in_seq
        else:
            out_seq_emb = self.embedding(out_seq)
            out_seq_emb_out, _ = self.decoder(out_seq_emb, (h, c))
            out_seq_vocab_logit = self.decoder2vocab(
                out_seq_emb_out)
            return out_seq_vocab_logit[:, :-1], out_seq[:-1]

    def forward(self, in_seq, out_seq=None, deterministic=False):
        mu, logvar = self.encode(in_seq)
        z = self.reparam(mu, logvar, deterministic=deterministic)
        out_seq_logit, _ = self.decode(
            z,
            out_seq,
            deterministic=deterministic)
        return out_seq_logit, mu, logvar

    def loss(self, in_seq, out_seq):
        out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        neg_likelihood = self.loss_func(
            out_seq_logit.transpose(1, 2),
            out_seq[:, 1:])
        neg_likelihood = neg_likelihood.sum(axis=1).mean()
        kl_div = -0.5 * (1.0 + logvar - mu ** 2
                         - torch.exp(logvar)).sum(axis=1).mean()
        return neg_likelihood + self.beta * kl_div

    def generate(self,
                 z=None,
                 sample_size=None,
                 deterministic=False):
        device = next(self.parameters()).device
        if z is None:
            z = torch.randn(sample_size,
                            self.latent_dim).to(device)
        else:
            z = z.to(device)
        with torch.no_grad():
            self.eval()
            _, out_seq = self.decode(z,
                                     deterministic=deterministic)
            out = [self.vocab.seq2smiles(each_seq)
                   for each_seq in out_seq]
            self.train()
            return out

    def reconstruct(self,
                    in_seq,
                    deterministic=True,
                    max_reconstruct=None,
                    verbose=True):
        self.eval()
        if max_reconstruct is not None:
            in_seq = in_seq[:max_reconstruct]
        mu, logvar = self.encode(in_seq)
        z = self.reparam(mu, logvar, deterministic=deterministic)
        _, out_seq = self.decode(z, deterministic=deterministic)

        success_list = []
        for each_idx, each_seq in enumerate(in_seq):
            truth = self.vocab.seq2smiles(each_seq)[::-1]
            pred = self.vocab.seq2smiles(out_seq[each_idx])
            success_list.append(truth==pred)
            if verbose:
                print('{}\t{} -> {}'.format(
                    truth==pred, truth, pred))
        self.train()
        return success_list


def trainer(
        model,
        train_tensor,
        val_tensor,
        smiles_vocab,
        n_epoch=10,
        lr=1e-3,
        batch_size=256,
        beta_schedule=[0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
        print_freq=100,
        device='cuda'):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(
        torch.flip(train_tensor, dims=[1]),
        train_tensor)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    val_dataset = TensorDataset(torch.flip(val_tensor, dims=[1]),
                                val_tensor)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    train_loss_list = []
    val_loss_list = []
    val_reconstruct_rate_list = []
    running_loss = 0
    running_sample_size = 0
    each_batch_idx = 0
    for each_epoch in range(n_epoch):
        try:
            model.beta = beta_schedule[each_epoch]
        except:
            pass
        print(' beta = {}'.format(model.beta))
        for each_train_batch in tqdm(train_data_loader):
            model.train()
            each_loss = model.loss(each_train_batch[0].to(device),
                                   each_train_batch[1].to(device))
            running_loss += each_loss.item()
            running_sample_size += len(each_train_batch[0])
            optimizer.zero_grad()
            each_loss.backward()
            optimizer.step()
            if (each_batch_idx+1) % print_freq == 0:
                train_loss_list.append((
                    each_batch_idx+1,
                    running_loss/running_sample_size))
                print('#epoch: {}\t#update: {},\tper-example '
                      'train loss:\t{}'.format(
                          each_epoch,
                          each_batch_idx+1,
                          running_loss/running_sample_size))
            running_loss = 0
            running_sample_size = 0
            each_batch_idx += 1
        val_loss = 0
        each_val_success_list = []
        with torch.no_grad():
            for each_val_batch in val_data_loader:
                val_loss += model.loss(
                    each_val_batch[0].to(device),
                    each_val_batch[1].to(device)).item()
                each_val_success_list.extend(model.reconstruct(
                    each_val_batch[0].to(device),
                    verbose=False))
        val_loss_list.append((each_batch_idx+1,
                              val_loss/len(val_dataset)))
        val_reconstruct_rate_list.append((
            each_batch_idx+1,
            sum(each_val_success_list)/len(each_val_success_list)
        ))
        print('#update: {},\tper-example val loss:\t{}'.format(
            each_batch_idx+1, val_loss/len(val_dataset)))
        print(' * reconstruction success rate: {}'.format(
            val_reconstruct_rate_list[-1][1]))

    return (train_loss_list,
            val_loss_list,
            val_reconstruct_rate_list)
