import itertools
import numba
import pdb
from collections import defaultdict
import seaborn as sb
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union, Optional
from torch_geometric.utils import to_dense_adj, to_networkx
from scipy.sparse.csgraph import floyd_warshall

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def create_2D_mol_from_3D_mol(mol):
    rdkit_mol = Chem.Mol()
    editable_mol = Chem.RWMol(rdkit_mol)

    for atom in mol.GetAtoms():
        editable_mol.AddAtom(atom)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        editable_mol.AddBond(i, j, bond_type)
    mol = editable_mol.GetMol()

    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    return mol


def extract_MMFF_energy_pos(rdkit_mol, num_conformers=5):
    try:
        mol = rdkit_mol
        result_list = AllChem.EmbedMultipleConfs(mol, num_conformers)
        result_list = AllChem.MMFFOptimizeMoleculeConfs(
            mol, mmffVariant="MMFF94s", numThreads=8
        )
        energy_list = [x[1] for x in result_list]
        index = np.argmin(energy_list)
        energy = energy_list[index]
        conformer = mol.GetConformer(id=int(index))
    except Exception as e:
        print(str(e))
        print("======bad")
        mol = rdkit_mol
        AllChem.Compute2DCoords(mol)
        energy = 0
        conformer = mol.GetConformer()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_feature = atomic_number - 1
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    MMFF_data = {"x": x, "energy": energy, "positions": positions}
    return MMFF_data


# note this is different from the 2D case
# For 2D, please refer to https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
atom_type_count = 119


def mol_to_graph_data_obj_simple_2D(mol, max_num_nodes=50):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atomic_number = atom.GetAtomicNum()
        assert atomic_number - 1 == atom_feature[0]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    # num_bond_features = 2  # bond type, bond direction
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
        edge_index=edge_index,
        edge_attr=edge_attr,
        positions=torch.empty((0, 3), dtype=torch.float),
        bond_lengths=torch.empty((0, 1), dtype=torch.float),
        bond_angles=torch.empty((0, 4), dtype=torch.float),
        # angle_directions=torch.empty((0,), dtype=torch.long),
        dihedral_angles=torch.empty((0, 5), dtype=torch.float),
        spd_mat=torch.empty((0), dtype=torch.long),
        num_angles=torch.tensor(0, dtype=torch.long),
        num_dihedrals=torch.tensor(0, dtype=torch.long),
    )
    return data


def get_bond_angles_rdkit(mol, edges_list, efficient=False):
    conformer = mol.GetConformers()[0]
    bond_angles = []

    group_1 = []  # holds angles in the interval [, 0.2]
    group_2 = []  # holds angles in the interval [0.5, ]
    group_3 = []  # everything else

    for i, edge in enumerate(edges_list):
        src, dst = edge
        for j in range(i + 1, len(edges_list)):
            src2, dst2 = edges_list[j]

            if src == dst2 and dst == src2:
                continue

            if src2 not in [src, dst] and dst2 not in [src, dst]:
                continue

            if src2 in [src, dst]:
                if src2 == src:
                    src, anchor, dst = dst, src, dst2
                elif src2 == dst:
                    src, anchor, dst = src, dst, dst2
            elif dst2 in [src, dst]:
                if dst2 == src:
                    src, anchor, dst = dst, src, src2
                elif dst2 == dst:
                    src, anchor, dst = src, dst, src2

            angle = (
                Chem.rdMolTransforms.GetAngleRad(conformer, src, anchor, dst) / np.pi
            )

            if efficient:
                if angle <= 0.2:
                    group_1.append([anchor, src, dst, angle])
                elif angle >= 0.5:
                    group_2.append([anchor, src, dst, angle])
                else:
                    group_3.append([anchor, src, dst, angle])
            else:
                bond_angles.append([anchor, src, dst, angle])

    if efficient:
        bond_angles.extend(group_3)
        np.random.shuffle(group_1)
        np.random.shuffle(group_2)
        group_1 = group_1[: len(group_1) // 2]
        group_2 = group_2[: len(group_2) // 2]

        bond_angles.extend(group_1)
        bond_angles.extend(group_2)

        np.random.shuffle(bond_angles)

    bond_angles = torch.tensor(bond_angles, dtype=torch.float32)
    perm = torch.randperm(bond_angles.shape[0])
    bond_angles = bond_angles[perm]
    return bond_angles


@numba.njit
def compute_angle(anchor, vec1, vec2):
    vec1 = vec1 - anchor
    vec2 = vec2 - anchor

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = np.arccos(np.dot(vec1, vec2))
    return angle


def get_angles(edges, atom_poses, dir_type="HT", get_complement_angles=False):
    # if there are E' bonds, then this will be E' x 3 where each row has 3 entries:
    # anchor, src, tar which are all node indices
    bond_angle_indices = []

    E = len(edges.T)
    edge_indices = np.arange(E)
    bond_angles = []
    bond_angle_dirs = []
    for tar_edge_i in range(E):
        tar_edge = edges.T[tar_edge_i]
        if dir_type == "HT":
            src_edge_indices = edge_indices[edges.T[:, 1] == tar_edge[0]]
        elif dir_type == "HH":
            src_edge_indices = edge_indices[edges.T[:, 1] == tar_edge[1]]
        else:
            raise ValueError(dir_type)
        for src_edge_i in src_edge_indices:
            if src_edge_i == tar_edge_i:
                continue
            src_edge = edges.T[src_edge_i]

            if dir_type == "HT":
                angle = compute_angle(
                    atom_poses[tar_edge[0]],
                    atom_poses[src_edge[1]],
                    atom_poses[tar_edge[1]],
                )
            elif dir_type == "HH":
                angle = compute_angle(
                    atom_poses[tar_edge[1]],
                    atom_poses[src_edge[0]],
                    atom_poses[tar_edge[0]],
                )
                angle /= np.pi  # normalize to [0, 1]

                if get_complement_angles:
                    complement_angle = compute_angle(
                        atom_poses[tar_edge[1]],
                        atom_poses[tar_edge[0]],
                        atom_poses[src_edge[0]],
                    )
                    complement_angle /= np.pi

            bond_angles.append(angle)
            if get_complement_angles:
                bond_angles.append(complement_angle)

            bond_angle_dirs.append(bool(src_edge[1] == tar_edge[0]))  # H -> H or H -> T
            if get_complement_angles:
                bond_angle_dirs.append(
                    bool(src_edge[0] == tar_edge[1])
                )  # H -> H or T -> H

            if dir_type == "HT":
                bond_angle_indices.append([tar_edge[0], src_edge[1], tar_edge[1]])
                if get_complement_angles:
                    bond_angle_indices.append([tar_edge[1], tar_edge[0], src_edge[0]])
            elif dir_type == "HH":
                bond_angle_indices.append([tar_edge[1], src_edge[0], tar_edge[0]])
                if get_complement_angles:
                    bond_angle_indices.append([tar_edge[1], tar_edge[0], src_edge[0]])

    bond_angle_indices = torch.tensor(bond_angle_indices)
    bond_angles = torch.tensor(bond_angles, dtype=torch.float32)

    bond_angles = torch.cat([bond_angle_indices, bond_angles.unsqueeze(-1)], dim=1)
    bond_angle_dirs = torch.tensor(bond_angle_dirs, dtype=torch.long)

    return bond_angles, bond_angle_dirs


def getTorsion(mol, bond):
    torsion_angles = []

    for atom in mol.GetAtomWithIdx(bond[0]).GetNeighbors():
        idx = atom.GetIdx()
        if idx != bond[1]:
            first = idx
            break

    for atom in mol.GetAtomWithIdx(bond[1]).GetNeighbors():
        idx = atom.GetIdx()
        if idx != bond[0]:
            last = idx
            break

    rotatable_bond = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    )

    i, j, k, l = first, bond[0], bond[1], last

    for bond in rotatable_bond:
        angle = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), i, j, k, l)
        torsion_angles.append([i, j, k, l, angle])

    torsion_angles = torch.tensor(torsion_angles, dtype=torch.float32)

    return torsion_angles


def enumerateTorsions(mol):
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = mol.GetSubstructMatches(torsionQuery)
    torsionList = []
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = mol.GetBondBetweenAtoms(idx2, idx3)
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)
        if (
            (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
            and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
        ) or (
            (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
            and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
        ):
            continue
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                torsionList.append((idx1, idx2, idx3, idx4))
    return torsionList


def get_eig_centrality(edge_index, num_nodes):
    nx_graph = to_networkx(Data(edge_index=edge_index), to_undirected=True)
    centrality = nx.eigenvector_centrality(nx_graph, max_iter=5000)
    centrality = torch.tensor(
        [centrality[i] for i in range(num_nodes)], dtype=torch.float32
    )
    return centrality

def get_betweenness_centrality(edge_index, num_nodes):
    nx_graph = to_networkx(Data(edge_index=edge_index), to_undirected=True)
    centrality = nx.betweenness_centrality(nx_graph)
    centrality = torch.tensor(
        [centrality[i] for i in range(num_nodes)], dtype=torch.float32
    )
    return centrality


def get_dihedral_angles(mol, bond_angles, edge_set, efficient=False):
    conformer = mol.GetConformers()[0]
    dihedral_angles = []

    angle_indices = bond_angles[:, :3].long()
    seen = set()

    for i in range(len(angle_indices)):
        anchor, head, tail = angle_indices[i]
        anchor = anchor.item()
        head = head.item()
        tail = tail.item()

        for edge in edge_set:
            if anchor in edge:
                continue
            if head in edge and tail in edge:
                continue
            e1, e2 = edge

            new_head, new_tail = None, None

            if head == e1:
                new_head = e2
            elif head == e2:
                new_head = e1

            if tail == e1:
                new_tail = e2
            elif tail == e2:
                new_tail = e1

            if new_head is not None:
                i, j, k, l = new_head, head, anchor, tail
                if i == j or k == l:
                    continue
                if (i, j, k, l) in seen:
                    continue
                angle_forward = (
                    Chem.rdMolTransforms.GetDihedralRad(conformer, i, j, k, l) / np.pi
                )
                angle_backward = (
                    Chem.rdMolTransforms.GetDihedralRad(conformer, l, k, j, i) / np.pi
                )
                # seen.add((i, j, k, l))

                # # # complement angle
                # seen.add((i, k, j, l))
                # seen.add((l, j, k, i))
                # seen.add((i, l, k, j))

                dihedral_angles.append([i, j, k, l, angle_forward])
                dihedral_angles.append([l, k, j, i, angle_backward])

                combinations = [
                    tuple(x) for x in itertools.permutations([i, j, k, l], r=4)
                ]
                seen.update(combinations)

            elif new_tail is not None:
                i, j, k, l = head, anchor, tail, new_tail
                if i == j or k == l:
                    continue
                if (i, j, k, l) in seen:
                    continue
                angle_forward = (
                    Chem.rdMolTransforms.GetDihedralRad(conformer, i, j, k, l) / np.pi
                )
                angle_backward = (
                    Chem.rdMolTransforms.GetDihedralRad(conformer, l, k, j, i) / np.pi
                )
                # seen.add((i, j, k, l))

                # # # complement angle
                # seen.add((i, k, j, l))
                # seen.add((l, j, k, i))
                # seen.add((i, l, k, j))

                dihedral_angles.append([i, j, k, l, angle_forward])
                dihedral_angles.append([l, k, j, i, angle_backward])

                combinations = [
                    tuple(x) for x in itertools.permutations([i, j, k, l], r=4)
                ]
                seen.update(combinations)

    dihedral_angles = torch.tensor(dihedral_angles, dtype=torch.float32)
    perm = torch.randperm(dihedral_angles.shape[0])
    dihedral_angles = dihedral_angles[perm]

    return dihedral_angles

def mol_to_graph_data_obj_just_data_3D(
    mol,
    pure_atomic_num=False,
):
    # atoms
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
    spd_mat = torch.zeros((1,), dtype=torch.long)
    # bonds
    # num_bond_features = 2  # bond type, bond direction
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

def mol_to_graph_data_obj_simple_3D(
    mol,
    pure_atomic_num=False,
    get_complement_angles=False,
    efficient=False,
    pretraining=False,
):
    # atoms
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
    spd_mat = torch.zeros((1,), dtype=torch.long)
    # bonds
    # num_bond_features = 2  # bond type, bond direction
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

        if pretraining:
            data = Data(
                x=x,
                positions=positions,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            graph = to_networkx(data).to_undirected()
            n_connected_components = len(list(nx.connected_components(graph)))
            if n_connected_components > 1:
                return None, None, "disconnected"

        bond_positions = positions[edge_index[0]] - positions[edge_index[1]]
        bond_lengths = torch.norm(bond_positions, dim=1).reshape(-1, 1)

        try:
            bond_angles = get_bond_angles_rdkit(mol, edges_list, efficient=efficient)
            num_angles = bond_angles.shape[0]
        except Exception as e:
            print(e)
            pdb.set_trace()

        if bond_angles.numel() == 0:
            bond_angles = torch.tensor([[0, 1, 0, np.pi]], dtype=torch.float32)
            num_angles = 1

        if bond_angles.numel() != 0:
            try:
                dihedral_angles = get_dihedral_angles(
                    mol,
                    bond_angles,
                    set(edges_list),
                    efficient=efficient,
                )
                num_dihedrals = dihedral_angles.shape[0]
            except Exception as e:
                print(e)
                pdb.set_trace()

            if dihedral_angles.numel() == 0:
                dihedral_angles = torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32)
                num_dihedrals = 1

        try:
            adj_matrix = to_dense_adj(edge_index).squeeze(0).long().cpu().numpy()
            spd_mat = floyd_warshall(adj_matrix, return_predecessors=False)
            spd_mat[spd_mat == np.inf] = 63  # max distance index

            spd_mat = torch.tensor(spd_mat).long()
            try:
                assert spd_mat.numel() != 0
            except:
                print("Empty spd_mat")
                pdb.set_trace()
        except Exception as e:
            print(e)
            pdb.set_trace()

        try:
            eig_centrality_vec = get_eig_centrality(edge_index, x.shape[0])
            betweenness_centrality_vec = get_betweenness_centrality(edge_index, x.shape[0])
        except Exception as e:
            print(e)
            pdb.set_trace()

    else:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        bond_lengths = torch.empty((0, 1), dtype=torch.float)
        bond_angles = torch.empty((0, 4), dtype=torch.float)
        # angle_directions = torch.empty((0, 1), dtype=torch.long)
        # angle_directions = torch.empty((0,), dtype=torch.long)
        dihedral_angles = torch.empty((0, 5), dtype=torch.float)
        spd_mat = torch.zeros((1,), dtype=torch.long)
        num_angles = 0
        num_dihedrals = 0

        return None, None, "no bonds"

    # assert that spd_mat is non-empty
    try:
        assert spd_mat.numel() != 0
    except Exception as e:
        print(e)
        pdb.set_trace()

    # pad spd_mat to 50 x 50
    # if spd_mat.shape[0] < 50:
    #     pad = 50 - spd_mat.shape[0]
    #     spd_mat = F.pad(spd_mat, (0, pad, 0, pad), value=63)

    num_angles = torch.tensor(num_angles, dtype=torch.long)
    num_dihedrals = torch.tensor(num_dihedrals, dtype=torch.long)

    data = Data(
        x=x,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        # angle_directions=angle_directions,
        dihedral_angles=dihedral_angles,
        spd_mat=spd_mat.flatten(),
        num_angles=num_angles,
        num_dihedrals=num_dihedrals,
        eig_centrality=eig_centrality_vec,
        betweenness_centrality=betweenness_centrality_vec
    )
    return data, atom_count, "success"


def mol_to_graph_data_obj_MMFF_3D(rdkit_mol, num_conformers):
    try:
        N = len(rdkit_mol.GetAtoms())
        if N > 100:  # for sider
            raise Exception
        rdkit_mol = Chem.AddHs(rdkit_mol)
        mol = rdkit_mol
        result_list = AllChem.EmbedMultipleConfs(mol, num_conformers)
        result_list = AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)
        energy_list = [x[1] for x in result_list]
        index = np.argmin(energy_list)
        energy = energy_list[index]
        conformer = mol.GetConformer(id=int(index))
    except:
        print("======bad")
        mol = rdkit_mol
        AllChem.Compute2DCoords(mol)
        energy = 0
        conformer = mol.GetConformer()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
        atomic_number = atom.GetAtomicNum()
        assert atomic_number - 1 == atom_feature[0]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(
        x=x,
        positions=positions,
    )
    return data


# Credits to https://github.com/chao1224/GraphMVP/blob/main/src_regression/datasets_complete_feature/molecule_datasets.py#L62
def graph_data_obj_to_nx_simple(data):
    """torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object"""
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8],
        )
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(
                begin_idx,
                end_idx,
                e0=temp_feature[0],
                e1=temp_feature[1],
                e2=temp_feature[2],
            )

    return G


# Credits to https://github.com/chao1224/GraphMVP/blob/main/src_regression/datasets_complete_feature/molecule_datasets.py#L62
def nx_to_graph_data_obj_simple(G):
    """vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered."""

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [
            node["x0"],
            node["x1"],
            node["x2"],
            node["x3"],
            node["x4"],
            node["x5"],
            node["x6"],
            node["x7"],
            node["x8"],
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["e0"], edge["e1"], edge["e2"]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_shifted_cells_within_radius_cutoff(
    structure, cutoff=5.0, numerical_tol=1e-8, max_neighbours=None
):
    from pymatgen.optimization.neighbors import find_points_in_spheres

    lattice_matrix = np.ascontiguousarray(
        np.array(structure.lattice.matrix), dtype=float
    )
    pbc = np.array([1, 1, 1], dtype=int)

    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, shifted_cells, distance = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices
    neighbor_indices = neighbor_indices
    shifted_cells = shifted_cells
    distance = distance
    exclude_self = (center_indices != neighbor_indices) | (distance > numerical_tol)
    center_indices, neighbor_indices, shifted_cells, distance = (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        shifted_cells[exclude_self],
        distance[exclude_self],
    )
    indices = []
    for center_, neighbor_ in zip(center_indices, neighbor_indices):
        indices.append([center_, neighbor_])

    if max_neighbours is None:
        return indices, shifted_cells, distance

    # Do distance sorting for each of the center unit cell.
    center_lattice2distance = defaultdict(list)
    for indice, shifted_cell, dist in zip(indices, shifted_cells, distance):
        first_index, second_index = indice
        center_lattice2distance[first_index].append(dist)
    first_index_list = center_lattice2distance.keys()
    sorted_center_lattice2distance_threshold = {}
    for first_index in first_index_list:
        sorted_dist_list = sorted(center_lattice2distance[first_index])
        if len(sorted_dist_list) <= max_neighbours:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[-1]
        else:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[
                max_neighbours
            ]

    # Filter w.r.t. sorted_center_lattice2distance_threshold.
    count = {x: 0 for x in range(len(first_index_list))}
    neo_indices, neo_shifted_cells, neo_distance = [], [], []
    for indice, shifted_cell, dis in zip(indices, shifted_cells, distance):
        first_index = indice[0]
        if dis <= sorted_center_lattice2distance_threshold[first_index] + numerical_tol:
            neo_indices.append(indice)
            neo_shifted_cells.append(shifted_cell)
            neo_distance.append(dis)
            count[first_index] += 1
    return neo_indices, neo_shifted_cells, neo_distance


def get_shifted_cells_within_kNN_cutoff(
    structure, numerical_tol=1e-8, max_neighbours=None
):
    from pymatgen.optimization.neighbors import find_points_in_spheres

    cutoff = 25

    lattice_matrix = np.ascontiguousarray(
        np.array(structure.lattice.matrix), dtype=float
    )
    pbc = np.array([1, 1, 1], dtype=int)

    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, shifted_cells, distance = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices
    neighbor_indices = neighbor_indices
    shifted_cells = shifted_cells
    distance = distance
    exclude_self = (center_indices != neighbor_indices) | (distance > numerical_tol)
    center_indices, neighbor_indices, shifted_cells, distance = (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        shifted_cells[exclude_self],
        distance[exclude_self],
    )
    indices = []
    for center_, neighbor_ in zip(center_indices, neighbor_indices):
        indices.append([center_, neighbor_])

    # Do distance sorting for each of the center unit cell.
    center_lattice2distance = defaultdict(list)
    for indice, shifted_cell, dist in zip(indices, shifted_cells, distance):
        first_index, second_index = indice
        center_lattice2distance[first_index].append(dist)
    first_index_list = center_lattice2distance.keys()
    sorted_center_lattice2distance_threshold = {}
    for first_index in first_index_list:
        sorted_dist_list = sorted(center_lattice2distance[first_index])
        if len(sorted_dist_list) <= max_neighbours:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[-1]
        else:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[
                max_neighbours
            ]

    # Filter w.r.t. sorted_center_lattice2distance_threshold.
    count = {x: 0 for x in range(len(first_index_list))}
    neo_indices, neo_shifted_cells, neo_distance = [], [], []
    for indice, shifted_cell, dis in zip(indices, shifted_cells, distance):
        first_index = indice[0]
        if dis <= sorted_center_lattice2distance_threshold[first_index] + numerical_tol:
            neo_indices.append(indice)
            neo_shifted_cells.append(shifted_cell)
            neo_distance.append(dis)
            count[first_index] += 1
    return neo_indices, neo_shifted_cells, neo_distance


def get_shifted_cells_within_radius_cutoff_v2(
    coordinates: np.ndarray,
    lattice: np.ndarray,
    max_distance: Union[float, None] = 4.0,
    max_neighbours: Union[int, None] = None,
    self_loops: bool = False,
    exclusive: bool = True,
    limit_only_max_neighbours: bool = False,
    numerical_tol: float = 1e-8,
    manual_super_cell_radius: float = None,
    super_cell_tol_factor: float = 0.25,
) -> list:
    r"""Generate range connections for a primitive unit cell in a periodic lattice (vectorized).
    The function generates a supercell of required radius and computes connections of neighbouring nodes
    from the primitive centered unit cell. For :obj:`max_neighbours` the supercell radius is estimated based on
    the unit cell density. Always the smallest necessary supercell is generated based on :obj:`max_distance` and
    :obj:`max_neighbours`. If a supercell for radius :obj:`max_distance` should always be generated but limited by
    :obj:`max_neighbours`, you can set :obj:`limit_only_max_neighbours` to `True`.
    .. warning::
        All atoms should be projected back into the primitive unit cell before calculating the range connections.
    .. note::
        For periodic structure, setting :obj:`max_distance` and :obj:`max_neighbours` to `inf` would also lead
        to an infinite number of neighbours and connections. If :obj:`exclusive` is set to `False`, having either
        :obj:`max_distance` or :obj:`max_neighbours` set to `inf`, will result in an infinite number of neighbours.
        If set to `None`, :obj:`max_distance` or :obj:`max_neighbours` can selectively be ignored.
    Args:
        coordinates (np.ndarray): Coordinate of nodes in the central primitive unit cell.
        lattice (np.ndarray): Lattice matrix of real space lattice vectors of shape `(3, 3)`.
            The lattice vectors must be given in rows of the matrix!
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to 4.0.
        max_neighbours (int, optional): Maximum number of allowed neighbours for each central atom. Default is None.
        self_loops (bool, optional): Allow self-loops between the same central node. Defaults to False.
        exclusive (bool): Whether both distance and maximum neighbours must be fulfilled. Default is True.
        limit_only_max_neighbours (bool): Whether to only use :obj:`max_neighbours` to limit the number of neighbours
            but not use it to calculate super-cell. Requires :obj:`max_distance` to be not `None`.
            Can be used if the super-cell should be generated with certain :obj:`max_distance`. Default is False.
        numerical_tol  (float): Numerical tolerance for distance cut-off. Default is 1e-8.
        manual_super_cell_radius (float): Manual radius for supercell. This is otherwise automatically set by either
            :obj:`max_distance` or :obj:`max_neighbours` or both. For manual supercell only. Default is None.
        super_cell_tol_factor (float): Tolerance factor for supercell relative to unit cell size. Default is 0.25.
    Returns:
        list: [indices, images, dist]
    credit to https://github.com/aimat-lab/gcnn_keras/blob/1c056f9a3b2990a1adb176c2dcc58c86d2ff64cf/kgcnn/graph/methods/_geom.py#L172
    """
    # Require either max_distance or max_neighbours to be specified.
    if max_distance is None and max_neighbours is None:
        raise ValueError(
            "Need to specify either `max_distance` or `max_neighbours` or both."
        )

    # Here we set the lattice matrix, with lattice vectors in either columns or rows of the matrix.
    lattice_col = np.transpose(lattice)
    lattice_row = lattice

    # Index list for nodes. Enumerating the nodes in the central unit cell.
    node_index = np.expand_dims(np.arange(0, len(coordinates)), axis=1)  # Nx1

    # Diagonals, center, volume and density of unit cell based on lattice matrix.
    center_unit_cell = np.sum(lattice_row, axis=0, keepdims=True) / 2  # (1, 3)
    max_radius_cell = np.amax(
        np.sqrt(np.sum(np.square(lattice_row - center_unit_cell), axis=-1))
    )
    max_diameter_cell = 2 * max_radius_cell
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(node_index) / volume_unit_cell

    # Center cell distance. Compute the distance matrix separately for the central primitive unit cell.
    # Here one can check if self-loops (meaning loops between the nodes of the central cell) should be allowed.
    center_indices = np.indices((len(node_index), len(node_index)))
    center_indices = center_indices.transpose(np.append(np.arange(1, 3), 0))  # NxNx2
    center_dist = np.expand_dims(coordinates, axis=0) - np.expand_dims(
        coordinates, axis=1
    )  # NxNx3
    center_image = np.zeros(center_dist.shape, dtype="int")
    if not self_loops:

        def remove_self_loops(x):
            m = np.logical_not(np.eye(len(x), dtype="bool"))
            x_shape = np.array(x.shape)
            x_shape[1] -= 1
            return np.reshape(x[m], x_shape)

        center_indices = remove_self_loops(center_indices)
        center_image = remove_self_loops(center_image)
        center_dist = remove_self_loops(center_dist)

    # Check the maximum atomic distance, since in practice atoms may not be inside the unit cell. Although they SHOULD
    # be projected back into the cell.
    max_diameter_atom_pair = np.amax(center_dist) if len(coordinates) > 1 else 0.0
    max_distance_atom_origin = np.amax(np.sqrt(np.sum(np.square(coordinates), axis=-1)))

    # Mesh Grid list. For a list of indices bounding left and right make a list of a 3D mesh.
    # Function is used to pad image unit cells or their origin for super-cell.
    def mesh_grid_list(bound_left: np.array, bound_right: np.array) -> np.array:
        pos = [np.arange(i, j + 1, 1) for i, j in zip(bound_left, bound_right)]
        grid_list = np.array(np.meshgrid(*pos)).T.reshape(-1, 3)
        return grid_list

    # Estimated real-space radius for max_neighbours based on density and volume of a single unit cell.
    if max_neighbours is not None:
        estimated_nn_volume = (max_neighbours + len(node_index)) / density_unit_cell
        estimated_nn_radius = abs(float(np.cbrt(estimated_nn_volume / np.pi * 3 / 4)))
    else:
        estimated_nn_radius = None

    # Determine the required size of super-cell
    if manual_super_cell_radius is not None:
        super_cell_radius = abs(manual_super_cell_radius)
    elif max_distance is None:
        super_cell_radius = estimated_nn_radius
    elif max_neighbours is None or limit_only_max_neighbours:
        super_cell_radius = max_distance
    else:
        if exclusive:
            super_cell_radius = min(max_distance, estimated_nn_radius)
        else:
            super_cell_radius = max(max_distance, estimated_nn_radius)

    # Safety for super-cell radius. We add this distance to ensure that all atoms of the outer images are within the
    # actual cutoff distance requested.
    super_cell_tolerance = max(
        max_diameter_cell, max_diameter_atom_pair, max_distance_atom_origin
    )
    super_cell_tolerance *= 1.0 + super_cell_tol_factor

    # Bounding box of real space cube with edge length 2 or inner sphere of radius 1 transformed into index
    # space gives 'bounding_box_unit'. Simply upscale for radius of super-cell.
    # To account for node pairing within the unit cell we add 'max_diameter_cell'.
    bounding_box_unit = np.sum(np.abs(np.linalg.inv(lattice_col)), axis=1)
    bounding_box_index = bounding_box_unit * (super_cell_radius + super_cell_tolerance)
    bounding_box_index = np.ceil(bounding_box_index).astype("int")

    # Making grid for super-cell that repeats the unit cell for required indices in 'bounding_box_index'.
    # Remove [0, 0, 0] of center unit cell by hand.
    bounding_grid = mesh_grid_list(-bounding_box_index, bounding_box_index)
    bounding_grid = bounding_grid[
        np.logical_not(np.all(bounding_grid == np.array([[0, 0, 0]]), axis=-1))
    ]  # Remove center cell
    bounding_grid_real = np.dot(bounding_grid, lattice_row)

    # Check which centers are in the sphere of cutoff, since for non-rectangular lattice vectors, the parallelepiped
    # can be overshooting the required sphere. Better do this here, before computing coordinates of nodes.
    dist_centers = np.sqrt(np.sum(np.square(bounding_grid_real), axis=-1))
    mask_centers = dist_centers <= (
        super_cell_radius + super_cell_tolerance + abs(numerical_tol)
    )
    images = bounding_grid[mask_centers]
    shifts = bounding_grid_real[mask_centers]

    # Compute node coordinates of images and prepare indices for those nodes. For 'N' nodes per cell and 'C' images
    # (without the central unit cell), this will be (flatten) arrays of (N*C)x3.
    num_images = images.shape[0]
    images = np.expand_dims(images, axis=0)  # 1xCx3
    images = np.repeat(images, len(coordinates), axis=0)  # NxCx3
    coord_images = np.expand_dims(coordinates, axis=1) + shifts  # NxCx3
    coord_images = np.reshape(coord_images, (-1, 3))  # (N*C)x3
    images = np.reshape(images, (-1, 3))  # (N*C)x3
    indices = np.expand_dims(np.repeat(node_index, num_images), axis=-1)  # (N*C)x1

    # Make distance matrix of central cell to all image. This will have shape Nx(NxC).
    dist = np.expand_dims(coord_images, axis=0) - np.expand_dims(
        coordinates, axis=1
    )  # Nx(N*C)x3
    dist_indices = np.concatenate(
        [
            np.repeat(np.expand_dims(node_index, axis=1), len(indices), axis=1),
            np.repeat(np.expand_dims(indices, axis=0), len(node_index), axis=0),
        ],
        axis=-1,
    )  # Nx(N*C)x2
    dist_images = np.repeat(
        np.expand_dims(images, axis=0), len(node_index), axis=0
    )  # Nx(N*C)x3

    # Adding distance matrix of nodes within the central cell to the image distance matrix.
    # The resulting shape is then Nx(NxC+1).
    dist_indices = np.concatenate([center_indices, dist_indices], axis=1)  # Nx(N*C+1)x2
    dist_images = np.concatenate([center_image, dist_images], axis=1)  # Nx(N*C+1)x2
    dist = np.concatenate([center_dist, dist], axis=1)  # Nx(N*C+1)x3

    # Distance in real space.
    dist = np.sqrt(np.sum(np.square(dist), axis=-1))  # Nx(N*C+1)

    # Sorting the distance matrix. Indices and image information must be sorted accordingly.
    arg_sort = np.argsort(dist, axis=-1)
    dist_sort = np.take_along_axis(dist, arg_sort, axis=1)
    dist_indices_sort = np.take_along_axis(
        dist_indices,
        np.repeat(np.expand_dims(arg_sort, axis=2), dist_indices.shape[2], axis=2),
        axis=1,
    )
    dist_images_sort = np.take_along_axis(
        dist_images,
        np.repeat(np.expand_dims(arg_sort, axis=2), dist_images.shape[2], axis=2),
        axis=1,
    )

    # Select range connections based on distance cutoff and nearest neighbour limit. Uses masking.
    # Based on 'max_distance'.
    if max_distance is None:
        mask_distance = np.ones_like(dist_sort, dtype="bool")
    else:
        mask_distance = dist_sort <= max_distance + abs(numerical_tol)
    # Based on 'max_neighbours'.
    mask_neighbours = np.zeros_like(dist_sort, dtype="bool")
    if max_neighbours is None:
        max_neighbours = dist_sort.shape[-1]
    mask_neighbours[:, :max_neighbours] = True

    if exclusive:
        mask = np.logical_and(mask_neighbours, mask_distance)
    else:
        mask = np.logical_or(mask_neighbours, mask_distance)

    # Select nodes.
    out_dist = dist_sort[mask]
    out_images = dist_images_sort[mask]
    out_indices = dist_indices_sort[mask]

    return [out_indices, out_images, out_dist]


def preiodic_augmentation_with_lattice(
    atom_feature_list,
    positions_list,
    lattice,
    center_and_shifted_edge_index_list,
    shifted_cell_list,
    shifted_distance_list,
):
    augmentation_record = defaultdict(list)
    key_index_dict = defaultdict(int)
    range_num = len(center_and_shifted_edge_index_list)

    key_total_index = 0
    neo_edge_index_list, neo_edge_distance_list = [], []
    periodic_index_mapping_list = []

    for first_indice, (atom_num, positions) in enumerate(
        zip(atom_feature_list, positions_list)
    ):
        original_image = [0, 0, 0]
        first_key = "id: {}, image: [{}, {}, {}]".format(
            first_indice, original_image[0], original_image[1], original_image[2]
        )
        augmentation_record[first_key] = [
            atom_feature_list[first_indice],
            positions_list[first_indice],
        ]
        if first_key not in key_index_dict:
            key_index_dict[first_key] = key_total_index
            periodic_index_mapping_list.append(key_total_index)
            key_total_index += 1

    for range_idx in range(range_num):
        range_indice_two_nodes = center_and_shifted_edge_index_list[range_idx]
        range_image = list(shifted_cell_list[range_idx])
        shifted_distance = shifted_distance_list[range_idx]

        first_indice = range_indice_two_nodes[0]
        original_image = [0, 0, 0]
        first_key = "id: {}, image: [{}, {}, {}]".format(
            first_indice, original_image[0], original_image[1], original_image[2]
        )
        if first_key not in augmentation_record:
            augmentation_record[first_key] = [
                atom_feature_list[first_indice],
                positions_list[first_indice],
            ]

        lattice_shift = np.array([0.0, 0.0, 0.0])
        for direction_idx in range(3):
            if range_image[direction_idx] != 0:
                lattice_shift += lattice[direction_idx] * range_image[direction_idx]
        second_indice = range_indice_two_nodes[1]
        second_key = "id: {}, image: [{}, {}, {}]".format(
            second_indice, range_image[0], range_image[1], range_image[2]
        )
        if second_key not in augmentation_record:
            augmentation_record[second_key] = [
                atom_feature_list[second_indice],
                positions_list[second_indice] + lattice_shift,
            ]

        # Notice: first_key is already in
        if second_key not in key_index_dict:
            key_index_dict[second_key] = key_total_index
            periodic_index_mapping_list.append(second_indice)
            key_total_index += 1
        first_key_index = key_index_dict[first_key]
        second_key_index = key_index_dict[second_key]
        neo_edge_index_list.append([first_key_index, second_key_index])
        neo_edge_distance_list.append(shifted_distance)
        neo_edge_index_list.append([second_key_index, first_key_index])
        neo_edge_distance_list.append(shifted_distance)

        # Notice: only consider one direction to keep consistent with the input center_and_shifted_edge_index_list
        neo_edge_vector = (
            positions_list[first_indice] - positions_list[second_indice] - lattice_shift
        )
        neo_dist = np.linalg.norm(neo_edge_vector)
        assert np.abs(neo_dist - shifted_distance) < 1e-10

    neo_atom_feature_list, neo_positions_list = [], []
    for key, value in augmentation_record.items():
        neo_atom_feature_list.append(value[0])
        neo_positions_list.append(value[1])

    ##### only for debugging #####
    for range_idx in range(range_num):
        range_indice_two_nodes = center_and_shifted_edge_index_list[range_idx]
        range_image = list(shifted_cell_list[range_idx])
        shifted_distance = shifted_distance_list[range_idx]

        first_indice = range_indice_two_nodes[0]
        original_image = [0, 0, 0]
        first_key = "id: {}, image: [{}, {}, {}]".format(
            first_indice, original_image[0], original_image[1], original_image[2]
        )

        lattice_shift = np.array([0.0, 0.0, 0.0])
        for direction_idx in range(3):
            if range_image[direction_idx] != 0:
                lattice_shift += lattice[direction_idx] * range_image[direction_idx]
        second_indice = range_indice_two_nodes[1]
        second_key = "id: {}, image: [{}, {}, {}]".format(
            second_indice, range_image[0], range_image[1], range_image[2]
        )

        first_key_index = key_index_dict[first_key]
        second_key_index = key_index_dict[second_key]
        first_position = neo_positions_list[first_key_index]
        second_position = neo_positions_list[second_key_index]

        assert (
            np.linalg.norm(
                neo_positions_list[first_key_index] - positions_list[first_indice]
            )
            < 1e-10
        )
        assert (
            np.linalg.norm(
                neo_positions_list[second_key_index]
                - positions_list[second_indice]
                - lattice_shift
            )
            < 1e-10
        )

    neo_edge_index_list = np.array(neo_edge_index_list)
    neo_edge_index_list = neo_edge_index_list.T

    return (
        neo_atom_feature_list,
        neo_positions_list,
        neo_edge_index_list,
        neo_edge_distance_list,
        periodic_index_mapping_list,
    )


def make_edges_into_two_direction(
    center_and_shifted_edge_index_list, shifted_distance_list
):
    range_num = len(center_and_shifted_edge_index_list)

    neo_edge_index_list, neo_edge_distance_list = [], []
    for range_idx in range(range_num):
        first_indice, second_indice = center_and_shifted_edge_index_list[range_idx]
        neo_edge_index_list.append([first_indice, second_indice])
        neo_edge_index_list.append([second_indice, first_indice])

        distance = shifted_distance_list[range_idx]
        neo_edge_distance_list.append(distance)
        neo_edge_distance_list.append(distance)

    return neo_edge_index_list, neo_edge_distance_list


class PeriodicTable:
    def __init__(
        self,
        csv_path,
        normalize_atomic_mass=True,
        normalize_atomic_radius=True,
        normalize_electronegativity=True,
        normalize_ionization_energy=True,
        imputation_atomic_radius=209.46,  # mean value
        imputation_electronegativity=1.18,  # educated guess (based on neighbour elements)
        imputation_ionization_energy=8.0,
    ):  # mean value
        self.data = pd.read_csv(csv_path)
        self.data["AtomicRadius"].fillna(imputation_atomic_radius, inplace=True)
        # Pm, Eu, Tb, Yb are inside the mp_e_form dataset, but have no electronegativity value
        self.data["Electronegativity"].fillna(
            imputation_electronegativity, inplace=True
        )
        self.data["IonizationEnergy"].fillna(imputation_ionization_energy, inplace=True)
        if normalize_atomic_mass:
            self._normalize_column("AtomicMass")
        if normalize_atomic_radius:
            self._normalize_column("AtomicRadius")
        if normalize_electronegativity:
            self._normalize_column("Electronegativity")
        if normalize_ionization_energy:
            self._normalize_column("IonizationEnergy")

    def _normalize_column(self, column):
        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[
            column
        ].std()

    def get_symbol(self, z: Optional[int] = None):
        if z is None:
            return self.data["Symbol"].to_list()
        else:
            return self.data.loc[z - 1]["Symbol"]

    def get_atomic_mass(self, z: Optional[int] = None):
        if z is None:
            return self.data["AtomicMass"].to_list()
        else:
            return self.data.loc[z - 1]["AtomicMass"]

    def get_atomic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data["AtomicRadius"].to_list()
        else:
            return self.data.loc[z - 1]["AtomicRadius"]

    def get_electronegativity(self, z: Optional[int] = None):
        if z is None:
            return self.data["Electronegativity"].to_list()
        else:
            return self.data.loc[z - 1]["Electronegativity"]

    def get_ionization_energy(self, z: Optional[int] = None):
        if z is None:
            return self.data["IonizationEnergy"].to_list()
        else:
            return self.data.loc[z - 1]["IonizationEnergy"]

    def get_oxidation_states(self, z: Optional[int] = None):
        if z is None:
            return list(
                map(
                    self.parse_oxidation_state_string,
                    self.data["OxidationStates"].to_list(),
                )
            )
        else:
            oxidation_states = self.data.loc[z - 1]["OxidationStates"]
            return self.parse_oxidation_state_string(oxidation_states, encode=True)

    @staticmethod
    def parse_oxidation_state_string(s: str, encode: bool = True):
        if encode:
            oxidation_states = [0] * 14
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(","):
                oxidation_states[int(i) - 7] = 1
        else:
            oxidation_states = []
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(","):
                oxidation_states.append(int(i))
        return oxidation_states
