# Credit to Stanford's OGB library: https://github.com/snap-stanford/ogb
# Modified to accommodate virtual node and edge features

import torch

# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc", "virtual"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "misc",
        "virtual",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc", "virtual"],
    "possible_formal_charge_list": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        "misc",
        "virtual",
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc", "virtual"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc", "virtual"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "misc",
        "virtual",
    ],
    "possible_is_aromatic_list": [False, True, "virtual"],
    "possible_is_in_ring_list": [False, True, "virtual"],
    "possible_bond_type_list": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "misc",
        "virtual",
    ],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
        "virtual",
    ],
    "possible_is_conjugated_list": [False, True, "virtual"],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_feature_dims():
    return list(
        map(
            len,
            [
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_chirality_list"],
                allowable_features["possible_degree_list"],
                allowable_features["possible_formal_charge_list"],
                allowable_features["possible_numH_list"],
                allowable_features["possible_number_radical_e_list"],
                allowable_features["possible_hybridization_list"],
                allowable_features["possible_is_aromatic_list"],
                allowable_features["possible_is_in_ring_list"],
            ],
        )
    )


def get_bond_feature_dims():
    return list(
        map(
            len,
            [
                allowable_features["possible_bond_type_list"],
                allowable_features["possible_bond_stereo_list"],
                allowable_features["possible_is_conjugated_list"],
            ],
        )
    )


def atom_feature_vector_to_dict(atom_feature):
    [
        atomic_num_idx,
        chirality_idx,
        degree_idx,
        formal_charge_idx,
        num_h_idx,
        number_radical_e_idx,
        hybridization_idx,
        is_aromatic_idx,
        is_in_ring_idx,
    ] = atom_feature

    feature_dict = {
        "atomic_num": allowable_features["possible_atomic_num_list"][atomic_num_idx],
        "chirality": allowable_features["possible_chirality_list"][chirality_idx],
        "degree": allowable_features["possible_degree_list"][degree_idx],
        "formal_charge": allowable_features["possible_formal_charge_list"][
            formal_charge_idx
        ],
        "num_h": allowable_features["possible_numH_list"][num_h_idx],
        "num_rad_e": allowable_features["possible_number_radical_e_list"][
            number_radical_e_idx
        ],
        "hybridization": allowable_features["possible_hybridization_list"][
            hybridization_idx
        ],
        "is_aromatic": allowable_features["possible_is_aromatic_list"][is_aromatic_idx],
        "is_in_ring": allowable_features["possible_is_in_ring_list"][is_in_ring_idx],
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx, bond_stereo_idx, is_conjugated_idx] = bond_feature

    feature_dict = {
        "bond_type": allowable_features["possible_bond_type_list"][bond_type_idx],
        "bond_stereo": allowable_features["possible_bond_stereo_list"][bond_stereo_idx],
        "is_conjugated": allowable_features["possible_is_conjugated_list"][
            is_conjugated_idx
        ],
    }

    return feature_dict


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.full_atom_feature_dims = get_atom_feature_dims()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(self.full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(self.full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


if __name__ == "__main__":
    from loader import GraphClassificationPygDataset

    dataset = GraphClassificationPygDataset(name="tox21")
    atom_enc = AtomEncoder(100)
    bond_enc = BondEncoder(100)

    print(atom_enc(dataset[0].x))
    print(bond_enc(dataset[0].edge_attr))
