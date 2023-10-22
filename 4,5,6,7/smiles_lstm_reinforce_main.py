import gzip
import math
import matplotlib.pyplot as plt
import pandas as pd
from smiles_vocab import SmilesVocabulary
from smiles_lstm_reinforce import SmilesLSTM, trainer, rl_trainer
from metrics import plogp
import pickle
from rdkit import Chem
import torch
from torchdrug.data.molecule import PackedMolecule
from torchdrug.metrics import penalized_logP
from tqdm import tqdm
from rdkit import RDLogger

lg = RDLogger.logger()
RDLogger.DisableLog('*')
lg.setLevel(RDLogger.CRITICAL)

device = 'cuda'

def valid_ratio(smiles_list):
    n_success = 0
    success_list = []
    for each_smiles in smiles_list:
        try:
            smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(each_smiles))
            n_success += 1
            success_list.append(smiles)
        except:
            pass
    return n_success / len(smiles_list), success_list

if __name__ == '__main__':
    smiles_vocab = SmilesVocabulary()
    train_tensor, train_smiles_list \
        = smiles_vocab.batch_update_from_file('train.smi',
                                              return_smiles=True)
    val_tensor, val_smiles_list \
        = smiles_vocab.batch_update_from_file('val.smi',
                                              return_smiles=True)

    train_plogp_tensor = plogp(train_smiles_list,
                               'train_plogp.pklz')
    val_plogp_tensor = plogp(val_smiles_list,
                             'val_plogp.pklz')

    lstm = SmilesLSTM(smiles_vocab,
                      hidden_size=512,
                      n_layers=4)

    try:
        lstm.load_state_dict(torch.load('pretrained.pt'))
        print('load pretrained.pt')
    except:
        lstm, train_loss_list, val_loss_list = trainer(
            lstm,
            train_tensor,
            val_tensor,
            smiles_vocab,
            lr=1e-3,
            n_epoch=20,
            batch_size=128,
            print_freq=100,
            device=device)
        torch.save(lstm.state_dict(), 'pretrained.pt')
        plt.plot(*list(zip(*train_loss_list)), label='train loss')
        plt.plot(*list(zip(*val_loss_list)),
                 label='validation loss',
                 marker='*')
        plt.legend()
        plt.xlabel('# of updates')
        plt.ylabel('Loss function')
        plt.savefig('learning_curve.pdf')
        plt.clf()

    lstm, rl_train_loss_list, avg_reward_list = rl_trainer(
        lstm,
        train_tensor,
        train_plogp_tensor,
        smiles_vocab,
        n_epoch=1000,
        sample_size=128,
        batch_size=128,
        print_freq=100,
        device=device)

    plt.plot(*list(zip(*avg_reward_list)), marker='.')
    plt.xlabel('# of updates')
    plt.ylabel('Expected return')
    plt.savefig('rl_curve.pdf')

    smiles_list = lstm.generate(sample_size=1000)
    success_ratio, success_smiles_list = valid_ratio(smiles_list)
    print('success rate: {}'.format(success_ratio))

    if success_smiles_list:
        success_packed_dataset \
            = PackedMolecule.from_smiles(success_smiles_list)
        plogp_tensor = penalized_logP(success_packed_dataset)
        print(' * plogp mean = {}'.format(plogp_tensor.mean()))
        res_df = pd.DataFrame(zip(smiles_list,
                                  plogp_tensor.tolist()),
                              columns=['smiles', 'plogp'])
        with gzip.open('mol.pklz', 'wb') as f:
            pickle.dump(res_df, f)
