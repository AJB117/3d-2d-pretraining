import pdb
import os
import json
import torch
import os.path as osp
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data


def mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False):
    # atoms
    mol = Chem.AddHs(mol)
    atom_features_list = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        if not pure_atomic_num:
            atom_feature = atom_to_feature_vector(atom)
            atomic_number = atom.GetAtomicNum()
            assert atomic_number - 1 == atom_feature[0]
            atom_count[atomic_number] += 1
        else:
            atomic_number = atom.GetAtomicNum()
            atom_feature = atomic_number - 1
            atom_count[atomic_number] += 1
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data, atom_count


class Molecule3D(InMemoryDataset):
    def __init__(
        self,
        root="/home/patrick/3d-pretraining/dataset/Molecule3D/data",
        split="train",
        split_mode="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        args=None,
    ):
        assert split in ["train", "val", "test"]
        assert split_mode in ["random", "scaffold"]
        self.split_mode = split_mode
        self.root = root
        self.target_df = pd.read_csv(osp.join(self.raw_dir, "properties.csv"))

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.avg_degree = 0
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, "{}_{}.pt".format(split_mode, split))
        )

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self):
        name = ""
        return name

    @property
    def processed_file_names(self):
        return [
            "random_train.pt",
            "random_val.pt",
            "random_test.pt",
            "scaffold_train.pt",
            "scaffold_val.pt",
            "scaffold_test.pt",
        ]

    def download(self):
        pass

    def pre_process(self):
        data_list = []
        data_smiles_list = []
        sdf_paths = [
            osp.join(self.raw_dir, "combined_mols_0_to_1000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_1000000_to_2000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_2000000_to_3000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_3000000_to_3899647.sdf"),
        ]
        suppl_list = [
            Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths
        ]

        abs_idx = -1
        avg_degree = 0
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f"{i+1}/{len(sdf_paths)}"):
                abs_idx += 1
                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                data, _ = mol_to_graph_data_obj_simple_3D(mol)
                avg_degree += (data.edge_index.shape[1] / 2) / data.num_nodes
                data_list.append(data)
                data_smiles_list.append(smiles)

        self.avg_degree = avg_degree / len(data_list)

        return data_list, data_smiles_list

    def process(self):
        dir_ = os.path.join(self.root, "Molecule3D" + "_full", "processed")
        os.makedirs(dir_, exist_ok=True)
        print("dir: ", dir_)
        saver_path = os.path.join(dir_, "geometric_data_processed.pt")
        if not os.path.exists(saver_path):
            full_list, full_smiles_list = self.pre_process()
            index_list = np.arange(len(full_list))

            data_list = [self.get_data_prop(full_list, idx) for idx in index_list]
            print("Saving to {}.".format(saver_path))
            torch.save(self.collate(data_list), saver_path)

            data_smiles_series = pd.Series(full_smiles_list)
            saver_path = os.path.join(dir_, "smiles.csv")
            print("Saving to {}.".format(saver_path))
            data_smiles_series.to_csv(saver_path, index=False, header=False)
        else:
            # TODO: this is for fast preprocessing. will add loader later.
            # full_list, full_smiles_list = self.pre_process()
            full_list = torch.load(saver_path)
            full_smiles_list = pd.read_csv(
                os.path.join(dir_, "smiles.csv"), header=None
            )[0].tolist()

        print("len of full list: {}".format(len(full_list)))
        print("len of full smiles list: {}".format(len(full_smiles_list)))
        print("target_df:", self.target_df.shape)

        full_data, full_slices = full_list

        print("making processed files:", self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for m, split_mode in enumerate(["random", "scaffold"]):
            ind_path = osp.join(self.raw_dir, "{}_split_inds.json").format(split_mode)
            with open(ind_path, "r") as f:
                index_list = json.load(f)

            for s, split in enumerate(["train", "valid", "test"]):
                # data_list = [self.get_data_prop(full_list, idx) for idx in index_list[split]]
                data_list = []
                for idx in index_list[split]:
                    data = Data()
                    for key in full_data.keys:
                        item, slices = full_data[key], full_slices[key]
                        if torch.is_tensor(item):
                            sl = list(repeat(slice(None), item.dim()))
                            sl[full_data.__cat_dim__(key, item)] = slice(
                                slices[idx], slices[idx + 1]
                            )
                        else:
                            sl = slice(slices[idx], slices[idx + 1])
                        data[key] = item[sl]

                    data_list.append(data)

                data_smiles_list = [full_smiles_list[idx] for idx in index_list[split]]
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                data_smiles_series = pd.Series(data_smiles_list)
                saver_path = os.path.join(
                    self.processed_dir, "{}_{}_smiles.csv".format(split_mode, split)
                )
                print("Saving to {}.".format(saver_path))
                data_smiles_series.to_csv(saver_path, index=False, header=False)

                torch.save(self.collate(data_list), self.processed_paths[s + 3 * m])

        million = 1000000
        for sample_size in [1 * million]:
            dir_ = os.path.join(
                self.root, "Molecule3D" + "_{}".format(sample_size), "processed"
            )
            os.makedirs(dir_, exist_ok=True)
            print("dir_", dir_)

            index_list = np.arange(sample_size)
            data_list = [self.get_data_prop(full_list, idx) for idx in index_list]
            data_smiles_list = [full_smiles_list[idx] for idx in index_list]
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data_smiles_series = pd.Series(data_smiles_list)
            saver_path = os.path.join(dir_, "smiles.csv")
            print("Saving to {}.".format(saver_path))
            data_smiles_series.to_csv(saver_path, index=False, header=False)

            saver_path = os.path.join(dir_, "geometric_data_processed.pt")
            print("Saving to {}.".format(saver_path))
            torch.save(self.collate(data_list), saver_path)
        return

    def get_data_prop(self, full_list, abs_idx):
        data = full_list[abs_idx]
        data.y = torch.FloatTensor(self.target_df.iloc[abs_idx, 1:].values)
        return data

    def __repr__(self):
        return "{}({})".format(self.name, len(self))

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(
                    slices[idx], slices[idx + 1]
                )
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data
