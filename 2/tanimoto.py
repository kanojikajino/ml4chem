import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors \
    import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
caffeine = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
theophylline = Chem.MolFromSmiles('Cn1c2c(c(=O)n(c1=O)C)[nH]cn2')
fp_c = GetMorganFingerprintAsBitVect(caffeine,
                                     radius=2,
                                     nBits=2**11)
fp_t = GetMorganFingerprintAsBitVect(theophylline,
                                     radius=2,
                                     nBits=2**11)
print('Tanimoto similarity: {}'.format(
    DataStructs.FingerprintSimilarity(fp_c, fp_t)))
