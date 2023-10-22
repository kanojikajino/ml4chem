from rdkit import Chem
import torch
from torchdrug.data.molecule import PackedMolecule
from torchdrug.metrics import penalized_logP


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


def compute_plogp(smiles_list):
    filtered_smiles_list, fail_idx_list \
        = filter_valid(smiles_list)
    if not filtered_smiles_list:
        return -30.0 * torch.ones(len(smiles_list))
    packed_dataset = PackedMolecule.from_smiles(
        filtered_smiles_list)
    _plogp_tensor = penalized_logP(packed_dataset)
    plogp_tensor = torch.zeros(len(smiles_list),
                               dtype=torch.float)
    each_other_idx = 0
    for each_idx in range(len(plogp_tensor)):
        if each_idx in fail_idx_list:
            plogp_tensor[each_idx] = -30.0
        else:
            plogp_tensor[each_idx] = _plogp_tensor[each_other_idx]
            each_other_idx += 1
    return plogp_tensor

