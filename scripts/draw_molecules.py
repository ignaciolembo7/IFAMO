# Crear la molécula con un anillo de pentano, dos grupos metilo en el mismo vértice, un nitrógeno en el vértice adyacente y un enlace al oxígeno
#mol = Chem.MolFromSmiles("C[C]1(N([O])C(C)CC1)C")

from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem import Draw

# Crear la molécula de benceno
mol = Chem.MolFromSmiles("c1ccccc1")

# Obtener uno de los átomos de carbono del benceno
carbono = mol.GetAtomWithIdx(0)

# Agregar un átomo de hidrógeno al átomo de carbono
hidrogeno = Chem.Atom("H")
mol.AddAtom(hidrogeno)
mol.AddBond(carbono.GetIdx(), hidrogeno.GetIdx(), Chem.BondType.SINGLE)

# Dibujar la molécula utilizando RDKit
img = Draw.MolToImage(mol)

# Guardar la imagen como PNG
img.save("benceno_hidrogeno.png")