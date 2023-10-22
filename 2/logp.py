from rdkit import Chem
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
logp = Descriptors.MolLogP(mol)
print('logp = {}'.format(logp))
