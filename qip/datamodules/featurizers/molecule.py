from openeye import oechem
from admet_prediction.datamodules.featurizers.base import MolFeature, mol_feature_registry, safe_index
from admet_prediction.typing import rdMol, OEMolBase, MOLTYPE
from rdkit import Chem

# The functional group descriptors in RDkit.


@mol_feature_registry.register(name="molweight")
class GetMolecularWeight(MolFeature):
    def __call__(self, mol: MOLTYPE) -> float:
        if isinstance(mol, rdMol):
            return Chem.Descriptors.ExactMolWt(mol)
        else:
            return oechem.OECalculateMolecularWeight(mol)


