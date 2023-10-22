import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from smiles_vocab import SmilesVocabulary
from smiles_vae import SmilesVAE, trainer
from rdkit import Chem
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def valid_ratio(smiles_list):
    n_success = 0
    for each_smiles in smiles_list:
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(each_smiles))
            n_success += 1
        except:
            pass
    return n_success / len(smiles_list)

if __name__ == '__main__':
    smiles_vocab = SmilesVocabulary()
    train_tensor = smiles_vocab.batch_update_from_file(
        'train.smi')
    val_tensor = smiles_vocab.batch_update_from_file('val.smi')
    max_len = val_tensor.shape[1]

    vae = SmilesVAE(smiles_vocab,
                    latent_dim=64,
                    emb_dim=256,
                    encoder_params={'hidden_size': 512,
                                    'num_layers': 1,
                                    'bidirectional': False,
                                    'dropout': 0.},
                    decoder_params={'hidden_size': 512,
                                    'num_layers': 1,
                                    'dropout': 0.},
                    encoder2out_params={'out_dim_list': [256]},
                    max_len=max_len)
    train_loss_list, val_loss_list, val_reconstruct_rate_list \
        = trainer(
            vae,
            train_tensor,
            val_tensor,
            smiles_vocab,
            lr=1e-4,
            n_epoch=100,
            batch_size=256,
            beta_schedule=[0.1],
            print_freq=100,
            device='cuda')
    plt.plot(*list(zip(*train_loss_list)), label='train loss')
    plt.plot(*list(zip(*val_loss_list)),
             label='validation loss',
             marker='*')
    plt.legend()
    plt.xlabel('# of updates')
    plt.ylabel('Loss function')
    plt.savefig('smiles_vae_learning_curve.pdf')
    plt.clf()

    plt.plot(*list(zip(*val_reconstruct_rate_list)),
             label='reconstruction rate')
    plt.legend()
    plt.xlabel('# of updates')
    plt.ylabel('Reconstruction rate')
    plt.savefig('reconstruction_rate_curve.pdf')

    smiles_list = vae.generate(sample_size=1000,
                               deterministic=True)
    print('success rate: {}'.format(valid_ratio(smiles_list)))
    
    torch.save(vae.state_dict(), 'vae.pt')

    with open('vae_smiles.pkl', 'wb') as f:
        pickle.dump(smiles_list, f)
