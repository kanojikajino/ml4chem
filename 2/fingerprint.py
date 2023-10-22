import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors \
    import GetMorganFingerprintAsBitVect
mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2**11)
fp_tensor = torch.tensor(fp)
fp_idx_tensor = torch.tensor(fp.GetOnBits())
print('fp_tensor = {}'.format(fp_tensor))
print('shape = {}'.format(fp_tensor.shape))
print('non-zero indices: {}'.format(fp_idx_tensor))
