from tqdm import tqdm
import pdb
import os
from itertools import repeat

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.constants import physical_constants
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_just_data_3D


class MoleculeDatasetQM8(InMemoryDataset):
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm8.zip"
    raw_url1 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv"

    def __init__(
        self,
        root,
        dataset,
        task,
        rotation_transform=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        calculate_thermo=True,
        use_pure_atomic_num=True,
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
        self.use_pure_atomic_num = use_pure_atomic_num

        self.target_field = [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ]
        self.pd_target_field = [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ]

        self.task = task
        if self.task == "qm8":
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

        # self.task_indices = torch.tensor(
        #     [list(self.conversion.keys()).index(task) for task in self.target_field]
        # )
        # self.eV2meV = torch.tensor(
        #     [
        #         1.0 if list(self.conversion.values())[task_index] == 1.0 else 1000
        #         for task_index in self.task_indices
        #     ]
        # )

        super(MoleculeDatasetQM8, self).__init__(
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
        for key in self.data.keys():
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
            "qm8.sdf",
            "qm8.csv",
        ]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        download_url(self.raw_url1, self.raw_dir)
        return

    def process(self):
        df = pd.read_csv(self.raw_paths[1])
        df = df[self.pd_target_field]

        target = df.to_numpy()
        target = torch.tensor(target, dtype=torch.float)

        data_df = pd.read_csv(self.raw_paths[1])
        whole_smiles_list = data_df["smiles"].tolist()
        print("TODO\t", whole_smiles_list[:100])

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        print("suppl: {}\tsmiles_list: {}".format(len(suppl), len(whole_smiles_list)))

        data_list, data_smiles_list, data_name_list, idx, invalid_count = (
            [],
            [],
            [],
            0,
            0,
        )
        for i, mol in enumerate(tqdm(suppl)):
            data, atom_count = mol_to_graph_data_obj_just_data_3D(
                mol,
                pure_atomic_num=self.use_pure_atomic_num,
            )

            data.id = torch.tensor([idx])
            temp_y = target[i]

            data.y = temp_y

            name = mol.GetProp("_Name")
            smiles = whole_smiles_list[i]

            # TODO: need double-check this
            temp_mol = AllChem.MolFromSmiles(smiles)
            if temp_mol is None:
                print("Exception with (invalid mol)\t", i)
                invalid_count += 1
                continue

            data_smiles_list.append(smiles)
            data_name_list.append(name)
            data_list.append(data)
            idx += 1

        print(
            "mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}".format(
                idx - 1, len(data_smiles_list), len(set(data_smiles_list))
            )
        )
        print("{} invalid molecules".format(invalid_count))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # TODO: need double-check later, the smiles list are identical here?
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data_name_series = pd.Series(data_name_list)
        saver_path = os.path.join(self.processed_dir, "name.csv")
        print("saving to {}".format(saver_path))
        data_name_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
