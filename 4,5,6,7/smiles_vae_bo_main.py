import gzip
import pickle
import pandas as pd
import torch
import math
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import (standardize,
                                      normalize,
                                      unnormalize)
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import DataLoader, TensorDataset

from smiles_vocab import SmilesVocabulary
from smiles_vae import SmilesVAE
from metrics import filter_valid, compute_plogp

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def bo_dataset_construction(vae,
                            input_tensor,
                            smiles_list,
                            batch_size=128,
                            max_batch=10):
    dataloader = DataLoader(TensorDataset(input_tensor),
                            batch_size=batch_size,
                            shuffle=False)
    z_list = []
    plogp_list = []
    out_smiles_list = []
    for each_batch_idx, each_tensor in enumerate(dataloader):
        if each_batch_idx == max_batch:
            break
        smiles_sublist = smiles_list[
            batch_size * each_batch_idx
            : batch_size * (each_batch_idx+1)]
        with torch.no_grad():
            z, _ = vae.encode(each_tensor[0].to(vae.device))
        z_list.append(z.to('cpu').double())
        plogp_tensor = compute_plogp(smiles_sublist)
        plogp_list.append(plogp_tensor.double())
        out_smiles_list.extend(smiles_sublist)
    return (torch.cat(z_list),
            torch.cat(plogp_list),
            out_smiles_list)


def obj_func(z, vae):
    z = z.to(torch.float32)
    for _ in range(100):
        smiles_list = vae.generate(z, deterministic=False)
        success_list, failed_idx_list = filter_valid(smiles_list)
        if success_list:
            smiles_list = success_list[:1]
            break
    plogp_tensor = compute_plogp(smiles_list).double()
    return plogp_tensor, smiles_list

if __name__ == '__main__':
    smiles_vocab = SmilesVocabulary()
    train_tensor, train_smiles_list\
        = smiles_vocab.batch_update_from_file('train.smi',
                                              with_smiles=True)
    val_tensor, val_smiles_list \
        = smiles_vocab.batch_update_from_file('val.smi',
                                              with_smiles=True)
    max_len = train_tensor.shape[1]
    latent_dim = 64

    vae = SmilesVAE(smiles_vocab,
                    latent_dim=latent_dim,
                    emb_dim=256,
                    encoder_params={'hidden_size': 512,
                                    'num_layers': 1,
                                    'bidirectional': False,
                                    'dropout': 0.},
                    decoder_params={'hidden_size': 512,
                                    'num_layers': 1,
                                    'dropout': 0.},
                    encoder2out_params={'out_dim_list': [256]},
                    max_len=max_len).to('cuda')
    vae.load_state_dict(torch.load('vae.pt'))
    vae.eval()

    z_tensor, plogp_tensor, smiles_list = bo_dataset_construction(
        vae,
        train_tensor,
        train_smiles_list)
    n_trial = 500

    for each_trial in range(n_trial):
        standardized_y = standardize(plogp_tensor).reshape(-1, 1)
        bounds = torch.stack([z_tensor.min(dim=0)[0],
                              z_tensor.max(dim=0)[0]])
        normalized_X = normalize(z_tensor, bounds)
        gp = SingleTaskGP(normalized_X, standardized_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1)
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=torch.stack([torch.zeros(latent_dim),
                                torch.ones(latent_dim)]),
            q=1,
            num_restarts=5,
            raw_samples=10)
        unnormalized_candidate = unnormalize(candidate, bounds)

        plogp_val, each_smiles_list = obj_func(
            unnormalized_candidate, vae)
        z_tensor = torch.cat([z_tensor, unnormalized_candidate])
        plogp_tensor = torch.cat([plogp_tensor, plogp_val])
        smiles_list.extend(each_smiles_list)
        print(' * {}\t{}'.format(
            each_trial,
            plogp_val))

    plogp_tensor = plogp_tensor[-n_trial:]
    smiles_list = smiles_list[-n_trial:]
    _, ascending_idx_tensor = plogp_tensor.sort()

    print('plogp\tsmiles')
    out_dict_list = []
    for each_idx in ascending_idx_tensor.tolist()[::-1][:10]:
        print('{}\t{}'.format(plogp_tensor[each_idx],
                              smiles_list[each_idx]))
        out_dict_list.append({'smiles': smiles_list[each_idx],
                              'plogp': plogp_tensor[each_idx]})

    res_df = pd.DataFrame(out_dict_list)
    with gzip.open('smiles_vae_best_mol.pklz', 'wb') as f:
        pickle.dump(res_df, f)

    with gzip.open('smiles_vae_bo_full.pklz', 'wb') as f:
        pickle.dump((smiles_list, plogp_tensor), f)
