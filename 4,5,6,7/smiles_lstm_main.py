import matplotlib.pyplot as plt
from smiles_vocab import SmilesVocabulary
from smiles_lstm import SmilesLSTM, trainer
from rdkit import Chem

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
    train_tensor = smiles_vocab.batch_update_from_file('train.smi')
    val_tensor = smiles_vocab.batch_update_from_file('val.smi')

    lstm = SmilesLSTM(smiles_vocab,
                      hidden_size=512,
                      n_layers=3)
    lstm, train_loss_list, val_loss_list = trainer(
        lstm,
        train_tensor,
        val_tensor,
        smiles_vocab,
        lr=1e-3,
        n_epoch=1,
        batch_size=128,
        print_freq=100,
        device='cuda')
    plt.plot(*list(zip(*train_loss_list)), label='train loss')
    plt.plot(*list(zip(*val_loss_list)),
             label='validation loss',
             marker='*')
    plt.legend()
    plt.xlabel('# of updates')
    plt.ylabel('Loss function')
    plt.savefig('smiles_lstm_learning_curve.pdf')
    smiles_list = lstm.generate(sample_size=1000)
    print('success rate: {}'.format(valid_ratio(smiles_list)))
