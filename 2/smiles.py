from rdkit import Chem
from rdkit.Chem import Draw
mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
Draw.MolToFile(mol, 'caffeine.svg')
