from openeye import oechem
from admet_prediction.datamodules.featurizers.base import BondFeature, bond_feature_registry, safe_index
from admet_prediction.typing import rdBond, OEBondBase, BONDTYPE


@bond_feature_registry.register(name="bond_type")
class GetBondType(BondFeature):
    VALID_FEATURES = tuple(["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])

    def __call__(self, bond: BONDTYPE) -> int:
        if isinstance(bond, rdBond):
            return safe_index(self.VALID_FEATURES, str(bond.GetBondType()))
        else:
            # TODO: check this implementation is correct
            # rdkit Bondtype enums
            # UNSPECIFIED = 0,
            # SINGLE,
            # DOUBLE,
            # TRIPLE,
            # QUADRUPLE,
            # QUINTUPLE,
            # HEXTUPLE,
            # ONEANDAHALF,
            # TWOANDAHALF,
            # THREEANDAHALF,
            # FOURANDAHALF,
            # FIVEANDAHALF,
            # AROMATIC,
            # IONIC,
            # HYDROGEN,
            # THREECENTER,
            # DATIVEONE,  //!< one-electron dative (e.g. from a C in a Cp ring to a metal)
            # DATIVE,     //!< standard two-electron dative
            # DATIVEL,    //!< standard two-electron dative
            # DATIVER,    //!< standard two-electron dative
            # OTHER,
            # ZERO //!< Zero-order bond (from http://pubs.acs.org/doi/abs/10.1021/ci200488k)
            def _bond_map_fn(bond):
                if bond.IsAromatic():
                    return "AROMATIC"
                else:
                    return self.VALID_FEATURES[int(bond.GetOrder() - 1)] if bond.GetOrder() in (1, 2, 3) else "misc"

            return safe_index(self.VALID_FEATURES, _bond_map_fn(bond))


@bond_feature_registry.register(name="bond_stereo")
class GetBondStereo(BondFeature):
    VALID_FEATURES = tuple(
        [
            "STEREONONE",  # no special style
            "STEREOZ",  # Z double bond
            "STEREOE",  # E double bond
            "STEREOCIS",  # cis double bond
            "STEREOTRANS",  # trans double bond
            "STEREOANY",  # intentionally unspecified
        ]
    )

    def __call__(self, bond: BONDTYPE) -> int:
        if isinstance(bond, rdBond):
            return self.VALID_FEATURES.index(str(bond.GetStereo()))
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
            v = []
            for neigh in bond.GetBgn().GetAtoms():
                if neigh != bond.GetEnd():
                    v.append(neigh)
                    break
            for neigh in bond.GetEnd().GetAtoms():
                if neigh != bond.GetBgn():
                    v.append(neigh)
                    break
            # 'OEBondStereo_All', 'OEBondStereo_Cis', 'OEBondStereo_CisTrans',
            # 'OEBondStereo_DoubleEither', 'OEBondStereo_Hash', 'OEBondStereo_Trans',
            # 'OEBondStereo_Undefined', 'OEBondStereo_Wavy', 'OEBondStereo_Wedge'
            stereo = bond.GetStereo(v, oechem.OEBondStereo_CisTrans)
            return stereo


@bond_feature_registry.register(name="is_conjugated")
class IsBondConjugated(BondFeature):
    VALID_FEATURES = tuple([False, True])

    def __call__(self, bond: BONDTYPE) -> int:
        if isinstance(bond, rdBond):
            return self.VALID_FEATURES.index(bond.GetIsConjugated())
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")


@bond_feature_registry.register(name="order")
class BondOrder(BondFeature):
    def __call__(self, bond: BONDTYPE) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
        return bond.GetOrder()


@bond_feature_registry.register(name="inring")
class IsBondRingMembership(BondFeature):
    def __call__(self, bond: BONDTYPE) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
        return bond.IsInRing()


@bond_feature_registry.register(name="aromaticity")
class IsBondAromatic(BondFeature):
    def __call__(self, bond: BONDTYPE) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
        return bond.IsAromatic()


@bond_feature_registry.register(name="rotatable")
class IsBondRotatable(BondFeature):
    def __call__(self, bond: BONDTYPE) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
        return bond.IsRotor()


@bond_feature_registry.register(name="chirality")
class IsBondChiral(BondFeature):
    def __call__(self, bond: BONDTYPE) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(bond)}")
        return bond.IsChiral()
