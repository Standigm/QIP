"""
The label generator for the pretraining.
"""
import numpy as np
from rdkit import Chem
import pickle
from collections import Counter
from multiprocessing import Pool
import tqdm


def atom2vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output

BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']
def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()

    """
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval(f"bond.Get{bond_feature}")()
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'

def bond2vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output



class TorchVocab(object):
    """
    Defines the vocabulary for atoms/bonds in molecule.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<pad>', '<other>'), vocab_type='atom'):
        """

        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        :param vocab_type: 'atom': atom atom_vocab; 'bond': bond atom_vocab.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.other_index = 1
        self.pad_index = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        # if self.vectors != other.vectors:
        #    return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def mol_to_seq(self, mol, with_len=False):
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        if self.vocab_type == 'atom':
            seq = [self.stoi.get(atom2vocab(mol, atom), self.other_index) for i, atom in enumerate(mol.GetAtoms())]
        else:
            seq = [self.stoi.get(bond2vocab(mol, bond), self.other_index) for i, bond in enumerate(mol.GetBonds())]
        return (seq, len(seq)) if with_len else seq

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class MolVocab(TorchVocab):
    def __init__(self, smiles, max_size=None, min_freq=1, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')

        print("Building %s vocab from smiles: %d" % (self.vocab_type, len(smiles)))
        counter = Counter()

        for smi in tqdm.tqdm(smiles):
            mol = Chem.MolFromSmiles(smi)
            if self.vocab_type == 'atom':
                for _, atom in enumerate(mol.GetAtoms()):
                    v = atom2vocab(mol, atom)
                    counter[v] += 1
            else:
                for _, bond in enumerate(mol.GetBonds()):
                    v = bond2vocab(mol, bond)
                    counter[v] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    def __init__(self, file_path, max_size=None, min_freq=1, num_workers=1, total_lines=None, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        print("Building %s vocab from file: %s" % (self.vocab_type, file_path))

        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        if total_lines is None:
            def file_len(fname):
                f_len = 0
                with open(fname) as f:
                    for f_len, _ in enumerate(f):
                        pass
                return f_len + 1

            total_lines = file_len(file_path)

        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        callback = lambda a: pbar.update(batch)
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            # print("Start: %d, End: %d"%(start, end))
            res.append(pool.apply_async(MolVocab.read_smiles_from_file,
                                        args=(file_path, start, end, vocab_type,),
                                        callback=callback))
            # read_smiles_from_file(lock, file_path, start, end)
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        # print(counter)
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    @staticmethod
    def read_smiles_from_file(file_path, start, end, vocab_type):
        # print("start")
        smiles = open(file_path, "r")
        smiles.readline()
        sub_counter = Counter()
        for i, smi in enumerate(smiles):
            if i < start:
                continue
            if i >= end:
                break
            mol = Chem.MolFromSmiles(smi)
            if vocab_type == 'atom':
                for atom in mol.GetAtoms():
                    v = atom2vocab(mol, atom)
                    sub_counter[v] += 1
            else:
                for bond in mol.GetBonds():
                    v = bond2vocab(mol, bond)
                    sub_counter[v] += 1
        # print("end")
        return sub_counter

    @staticmethod
    def load_vocab(vocab_path: str) -> 'MolVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


"""
The dataset used in training GROVER.
"""
import math
import os
import csv
from typing import Union, List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem




class GroverCollator:
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab

    def atom_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on atoms.
        :param smiles_batch:
        :return: The corresponding atom labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom2vocab(mol, atom), self.atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def bond_random_mask(self, smiles_batch):
        """
        Perform the random mask operaiion on bonds.
        :param smiles_batch:
        :return: The corresponding bond labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            nm_atoms = mol.GetNumAtoms()
            nm_bonds = mol.GetNumBonds()
            mlabel = []
            n_mask = math.ceil(nm_bonds * percent)
            perm = np.random.permutation(nm_bonds)[:n_mask]
            virtual_bond_id = 0
            for a1 in range(nm_atoms):
                for a2 in range(a1 + 1, nm_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if virtual_bond_id in perm:
                        label = self.bond_vocab.stoi.get(bond2vocab(mol, bond), self.bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            # todo: might need to consider bond_drop_rate
            # todo: double check reverse bond
            vocab_label.extend(mlabel)
        return vocab_label

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        atom_vocab_label = torch.Tensor(self.atom_random_mask(smiles_batch)).long()
        bond_vocab_label = torch.Tensor(self.bond_random_mask(smiles_batch)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        # may be some mask here
        res = {"graph_input": batchgraph,
               "targets": {"av_task": atom_vocab_label,
                           "bv_task": bond_vocab_label,
                           "fg_task": fgroup_label}
               }
        return res