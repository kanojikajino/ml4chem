from smiles_vocab import SmilesVocabulary

smiles_vocab = SmilesVocabulary()
train_tensor = smiles_vocab.batch_update_from_file('train.smi')
print(train_tensor)
print(train_tensor.shape)
print(smiles_vocab.char_list)
print(smiles_vocab.seq2smiles(train_tensor[0]))
