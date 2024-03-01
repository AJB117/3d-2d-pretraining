import pickle
import pdb
import os
from itertools import repeat

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.constants import physical_constants
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D
from Geom3D.datasets.dataset_QM9 import MoleculeDatasetQM9


class GeneratedMoleculeDatasetQM9(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        task,
        split_name,
        rotation_transform=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        calculate_thermo=True,
    ):
        """
        The complete columns are
        A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom
        and we take
        mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv
        """
        self.root = root
        self.rotation_transform = rotation_transform
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.split_name = split_name

        self.target_field = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
            "gap_02",
        ]
        self.pd_target_field = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
        ]
        self.task = task
        if self.task == "qm9":
            self.task_id = None
        else:
            self.task_id = self.target_field.index(task)
        self.calculate_thermo = calculate_thermo
        self.atom_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        #  TODO: need double-check
        #  https://github.com/atomistic-machine-learning/SchNetpack/blob/master/src/SchNetpack/datasets/qm9.py
        # HAR2EV = 27.211386246
        # KCALMOL2EV = 0.04336414
        #
        # conversion = torch.tensor([
        #     1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
        #     1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
        # ])

        #  Now we are following these two:
        #  https://github.com/risilab/cormorant/blob/master/examples/train_qm9.py
        #  https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/QM9.py

        self.hartree2eV = physical_constants["hartree-electron volt relationship"][0]

        self.conversion = {
            "mu": 1.0,
            "alpha": 1.0,
            "homo": self.hartree2eV,
            "lumo": self.hartree2eV,
            "gap": self.hartree2eV,
            "gap_02": self.hartree2eV,
            "r2": 1.0,
            "zpve": self.hartree2eV,
            "u0": self.hartree2eV,
            "u298": self.hartree2eV,
            "h298": self.hartree2eV,
            "g298": self.hartree2eV,
            "cv": 1.0,
        }
        self.task_indices = torch.tensor(
            [list(self.conversion.keys()).index(task) for task in self.target_field]
        )
        self.eV2meV = torch.tensor(
            [
                1.0 if list(self.conversion.values())[task_index] == 1.0 else 1000
                for task_index in self.task_indices
            ]
        )

        super(GeneratedMoleculeDatasetQM9, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.dataset = dataset
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))

        return

    def mean(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.mean(dim=0)
        return y

    def std(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.std(dim=0)
        return y

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        if self.rotation_transform is not None:
            data.positions = self.rotation_transform(data.positions)
        return data

    @property
    def raw_file_names(self):
        return [
            f"{self.split_name}_confs",
        ]

    @property
    def processed_file_names(self):
        return f"{self.split_name}_gen_data_processed.pt"

    def download(self):
        pass

    def process(self):
        data_root = "data/molecule_datasets/{}".format("QM9")
        reference_dataset = MoleculeDatasetQM9(data_root, dataset="QM9", task=self.task)
        confs_tups = pickle.load(open(self.raw_paths[0], "rb")).values()
        confs = [c[0] for c in confs_tups]
        confs_idx = [c[1] for c in confs_tups]
        ground_truth = reference_dataset[[c+1 for c in confs_idx]]

        data_list = []
        for i, true_data in enumerate(ground_truth):
            data, atom_count = mol_to_graph_data_obj_simple_3D(
                confs[i], pure_atomic_num=True
            )

            data.id = true_data.id
            data.y = true_data.y
            data_list.append(data)

        print(f"number of conformers: {len(data_list)}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # TODO: need double-check later, the smiles list are identical here?
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
