import torch
from torch import nn

class SmilesVocabulary(object):

    pad = ' '
    sos = '!'
    eos = '?'
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    def __init__(self):
        self.char_list = [self.pad, self.sos, self.eos]

    def update(self, smiles):
        char_set = set(smiles)
        char_set = char_set - set(self.char_list)
        self.char_list.extend(sorted(list(char_set)))
        return self.smiles2seq(smiles)

    def smiles2seq(self, smiles):
        return torch.tensor(
            [self.sos_idx]
            + [self.char_list.index(each_char)
               for each_char in smiles]
            + [self.eos_idx])

    def seq2smiles(self, seq, wo_special_char=True):
        if wo_special_char:
            return self.seq2smiles(seq[torch.where(
                (seq != self.pad_idx)
                * (seq != self.sos_idx)
                * (seq != self.eos_idx))],
                                   wo_special_char=False)
        return ''.join([
            self.char_list[each_idx] for each_idx in seq])

    def batch_update(self, smiles_list):
        seq_list = []
        out_smiles_list = []
        for each_smiles in smiles_list:
            if each_smiles.endswith('\n'):
                each_smiles = each_smiles.strip()
            seq_list.append(self.update(each_smiles))
            out_smiles_list.append(each_smiles)
        right_padded_batch_seq = nn.utils.rnn.pad_sequence(
            seq_list,
            batch_first=True,
            padding_value= self.pad_idx)
        return right_padded_batch_seq, out_smiles_list

    def batch_update_from_file(
            self, file_path, with_smiles=False):
        seq_tensor, smiles_list = self.batch_update(
            open(file_path).readlines())
        if with_smiles:
            return seq_tensor, smiles_list
        return seq_tensor
