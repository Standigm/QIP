from qip.datamodules.featurizers.base import AtomFeature, atom_feature_registry, safe_index
from qip.typing import rdAtom, ATOMTYPE
from qip.utils.molecule import ATOMNUM2GROUP, ATOMNUM2PERIOD


@atom_feature_registry.register(name="atomic_num")
class GetAtomicNum(AtomFeature):
    VALID_FEATURES = tuple(list(range(1, 119)))

    def __call__(self, atom: ATOMTYPE) -> int:
        return safe_index(self.VALID_FEATURES, atom.GetAtomicNum())


@atom_feature_registry.register(name="atomic_group")
class GetAtomicGroup(AtomFeature):
    VALID_FEATURES = tuple(list(range(1, 19)))

    def __call__(self, atom: ATOMTYPE) -> int:
        return safe_index(self.VALID_FEATURES, ATOMNUM2GROUP[atom.GetAtomicNum()])


@atom_feature_registry.register(name="atomic_period")
class GetAtomicPeriod(AtomFeature):
    VALID_FEATURES = tuple(list(range(1, 7)))

    def __call__(self, atom: ATOMTYPE) -> int:
        """
        Returns the periodic table period(row) of the element.
        """
        return safe_index(self.VALID_FEATURES, ATOMNUM2PERIOD[atom.GetAtomicNum()])


@atom_feature_registry.register(name="atomic_family")
class GetAtomicFamily(AtomFeature):
    VALID_FEATURES = tuple(list(range(10)))

    def __call__(self, atom: ATOMTYPE) -> int:
        """
        Returns the periodic table family of the element.
        0: Other non metals
        1: Alkali metals
        2: Alkaline earth metals
        3: Transition metals
        4: Lanthanides
        5: Actinides
        6: Other metals
        7: Metalloids
        8: Halogens
        9: Noble gases
        """
        z = atom.GetAtomicNum()
        if z in [1, 6, 7, 8, 15, 16, 34]:
            return 0
        if z in [3, 11, 19, 37, 55, 87]:
            return 1
        if z in [4, 12, 20, 38, 56, 88]:
            return 2
        if z in range(57, 72):
            return 4
        if z in range(89, 104):
            return 5
        if z in [13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 166]:
            return 6
        if z in [5, 14, 32, 33, 51, 52]:
            return 7
        if z in [9, 17, 35, 53, 85, 117]:
            return 8
        if z in [2, 10, 18, 36, 54, 86, 118]:
            return 9
        else:
            return 3


@atom_feature_registry.register(name="chirality")
class GetAtomChiarlity(AtomFeature):
    VALID_FEATURES = tuple(["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])

    def __call__(self, atom: ATOMTYPE) -> bool:
        if isinstance(atom, rdAtom):
            return self.VALID_FEATURES.index(str(atom.GetChiralTag()))
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
 




@atom_feature_registry.register(name="degree")
class GetAtomDegree(AtomFeature):
    VALID_FEATURES = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            return safe_index(self.VALID_FEATURES, atom.GetTotalDegree())
        else:
            return safe_index(self.VALID_FEATURES, atom.GetDegree())


@atom_feature_registry.register(name="formal_charge")
class GetAtomFormalCharge(AtomFeature):
    VALID_FEATURES = tuple([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    def __call__(self, atom: ATOMTYPE) -> int:
        return safe_index(self.VALID_FEATURES, atom.GetFormalCharge())


@atom_feature_registry.register(name="numH")
class GetAtomNumH(AtomFeature):
    VALID_FEATURES = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8])

    def __call__(self, atom: rdAtom) -> int:
        if isinstance(atom, rdAtom):
            return safe_index(self.VALID_FEATURES, atom.GetTotalNumHs())
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")


@atom_feature_registry.register(name="num_radical_e")
class GetNumRadicalElectrons(AtomFeature):
    VALID_FEATURES = tuple([0, 1, 2, 3, 4])

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            return safe_index(self.VALID_FEATURES, atom.GetNumRadicalElectrons())
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")


@atom_feature_registry.register(name="hybridization")
class GetAtomHybridization(AtomFeature):
    VALID_FEATURES = tuple(["SP", "SP2", "SP3", "SP3D", "SP3D2"])

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            return safe_index(self.VALID_FEATURES, str(atom.GetHybridization()))
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
            return atom.GetHyb()


@atom_feature_registry.register(name="radius")
class GetAtomRadius(AtomFeature):
    VALID_FEATURES = ()

    def __call__(self, atom: ATOMTYPE) -> float:
        if isinstance(atom, rdAtom):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
            return atom.GetRadius()


@atom_feature_registry.register(name="is_aromatic")
class IsAtomAromatic(AtomFeature):
    VALID_FEATURES = tuple([False, True])

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            return self.VALID_FEATURES.index(atom.GetIsAromatic())
        else:
            return self.VALID_FEATURES.index(atom.IsAromatic())


@atom_feature_registry.register(name="is_in_ring")
class IsAtomRingMembership(AtomFeature):
    VALID_FEATURES = tuple([False, True])

    def __call__(self, atom: ATOMTYPE) -> int:
        return self.VALID_FEATURES.index(atom.IsInRing())


@atom_feature_registry.register(name="is_chirality")
class IsAtomChiarl(AtomFeature):
    VALID_FEATURES = tuple([False, True])

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
        else:
            return self.VALID_FEATURES.index(atom.IsChiral())


@atom_feature_registry.register(name="valence")
class GetAtomValence(AtomFeature):
    VALID_FEATURES = ()

    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
            return atom.GetValence()



@atom_feature_registry.register(name="implicitnumH")
class GetAtomImplicitHcount(AtomFeature):
    def __call__(self, atom: ATOMTYPE) -> int:
        if isinstance(atom, rdAtom):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(atom)}")
            return atom.GetImplicitHCount()
