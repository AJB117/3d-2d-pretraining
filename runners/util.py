import pdb
import random
import numpy as np
from scipy import stats
from math import sqrt

from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from torch import Tensor

import ase
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from Geom3D.models.encoders import get_atom_feature_dims, get_bond_feature_dims


def apply_init(initializer: str = "glorot_uniform"):
    if initializer == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif initializer == "glorot_normal":
        return nn.init.xavier_normal_
    elif initializer == "he_uniform":
        return nn.init.kaiming_uniform_
    elif initializer == "he_normal":
        return nn.init.kaiming_normal_
    elif initializer == "orthogonal":
        return nn.init.orthogonal_
    else:
        raise ValueError("Invalid initializer name.")


def CL_acc(x1, x2, pos_mask=None):
    batch_size, _ = x1.size()
    if (
        x1.shape != x2.shape and pos_mask is None
    ):  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
        x2 = x2[:batch_size]
    sim_matrix = torch.einsum("ik,jk->ij", x1, x2)

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = sim_matrix / torch.einsum("i,j->ij", x1_abs, x2_abs)

    preds = (sim_matrix + 1) / 2 > 0.5
    if pos_mask is None:  # if we are comparing global with global
        pos_mask = torch.eye(batch_size, device=x1.device)
    neg_mask = 1 - pos_mask

    num_positives = len(x1)
    num_negatives = len(x1) * (len(x2) - 1)
    true_positives = (
        num_positives - ((preds.long() - pos_mask) * pos_mask).count_nonzero()
    )
    true_negatives = (
        num_negatives - (((~preds).long() - neg_mask) * neg_mask).count_nonzero()
    )

    true_positive_rate = true_positives / num_positives
    true_negative_rate = true_negatives / num_negatives

    return (
        (true_positives / num_positives + true_negatives / num_negatives) / 2,
        true_positive_rate,
        true_negative_rate,
    )


def perturb(
    pos,
    rotation=None,
    translation=None,
    rot_mu=0,
    rot_sigma=0.1,
    trans_mu=0,
    trans_sigma=1,
    device="cuda",
):
    if rotation is None:
        random_matrix = torch.normal(
            rot_mu, rot_sigma, size=(pos.shape[1], pos.shape[1])
        )
        rotation, _ = torch.qr(random_matrix)

    if translation is None:
        translation = torch.normal(trans_mu, trans_sigma, size=(1, pos.shape[1]))

    rotation = rotation.to(device)
    translation = translation.to(device)

    rotated_pos = rotation @ pos.T
    translated_pos = pos + translation

    return (
        rotated_pos,
        translated_pos,
        rotation,
        translation,
    )  # return rotation and translation for debugging


# credit to 3D Infomax (https://github.com/HannesStark/3DInfomax) for the following code
def NTXentLoss(z1: torch.Tensor, z2: torch.Tensor, tau=0.1, norm=True):
    sim_matrix = torch.einsum("ik, jk->ij", z1, z2)

    if norm:
        z1 = z1.norm(dim=1)
        z2 = z2.norm(dim=1)
        sim_matrix = sim_matrix / (torch.einsum("i, j->ij", z1, z2) + 1e-8)

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix.diag()

    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()

    return loss


cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.CL_similarity_metric == "InfoNCE_dot_prod":
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1.0 / B

    elif args.CL_similarity_metric == "EBM_dot_prod":
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat(
            [Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0
        )
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + args.CL_neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / (
            len(pred_pos) + len(pred_neg)
        )
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_similarity_metric == "EBM_node_dot_prod":
        criterion = nn.BCEWithLogitsLoss()

        neg_index = torch.randperm(len(Y))
        neg_Y = torch.cat([Y[neg_index]])

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / (
            len(pred_pos) + len(pred_neg)
        )
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


# def compute_accuracy(pred, target):
#     return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


# def do_AttrMasking(batch, criterion, node_repr, molecule_atom_masking_model):
#     target = batch.mask_node_label[:, 0]
#     node_pred = molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
#     attributemask_loss = criterion(node_pred.double(), target)
#     attributemask_acc = compute_accuracy(node_pred, target)
#     return attributemask_loss, attributemask_acc


# def do_ContextPred(batch, criterion, args, molecule_substruct_model,
#                    molecule_context_model, molecule_readout_func):

#     # creating substructure representation
#     substruct_repr = molecule_substruct_model(
#         batch.x_substruct, batch.edge_index_substruct,
#         batch.edge_attr_substruct)[batch.center_substruct_idx]

#     # creating context representations
#     overlapped_node_repr = molecule_context_model(
#         batch.x_context, batch.edge_index_context,
#         batch.edge_attr_context)[batch.overlap_context_substruct_idx]

#     # positive context representation
#     # readout -> global_mean_pool by default
#     context_repr = molecule_readout_func(overlapped_node_repr,
#                                          batch.batch_overlapped_context)

#     # negative contexts are obtained by shifting
#     # the indices of context embeddings
#     neg_context_repr = torch.cat(
#         [context_repr[cycle_index(len(context_repr), i + 1)]
#          for i in range(args.contextpred_neg_samples)], dim=0)

#     num_neg = args.contextpred_neg_samples
#     pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
#     pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

#     loss_pos = criterion(pred_pos.double(),
#                          torch.ones(len(pred_pos)).to(pred_pos.device).double())
#     loss_neg = criterion(pred_neg.double(),
#                          torch.zeros(len(pred_neg)).to(pred_neg.device).double())

#     contextpred_loss = loss_pos + num_neg * loss_neg

#     num_pred = len(pred_pos) + len(pred_neg)
#     contextpred_acc = (torch.sum(pred_pos > 0).float() +
#                        torch.sum(pred_neg < 0).float()) / num_pred
#     contextpred_acc = contextpred_acc.detach().cpu().item()

#     return contextpred_loss, contextpred_acc


# def check_same_molecules(s1, s2):
#     mol1 = AllChem.MolFromSmiles(s1)
#     mol2 = AllChem.MolFromSmiles(s2)
#     return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


# def rmse(y, f):
#     return sqrt(((y - f) ** 2).mean(axis=0))


# def mse(y, f):
#     return ((y - f) ** 2).mean(axis=0)


# def pearson(y, f):
#     return np.corrcoef(y, f)[0, 1]


# def spearman(y, f):
#     return stats.spearmanr(y, f)[0]


# def ci(y, f):
#     ind = np.argsort(y)
#     y = y[ind]
#     f = f[ind]
#     i = len(y) - 1
#     j = i - 1
#     z = 0.0
#     S = 0.0
#     while i > 0:
#         while j >= 0:
#             if y[i] > y[j]:
#                 z = z + 1
#                 u = f[i] - f[j]
#                 if u > 0:
#                     S = S + 1
#                 elif u == 0:
#                     S = S + 0.5
#             j = j - 1
#         i = i - 1
#         j = i - 1
#     # ci = S / z
#     return S / z


def get_num_task(dataset):
    """used in molecule_finetune.py"""
    if dataset == "tox21":
        return 12
    elif dataset in ["hiv", "bace", "bbbp", "donor"]:
        return 1
    elif dataset == "pcba":
        return 92
    elif dataset == "muv":
        return 17
    elif dataset == "toxcast":
        return 617
    elif dataset == "sider":
        return 27
    elif dataset == "clintox":
        return 2
    raise ValueError("Invalid dataset name.")


# Credit to PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/virtual_node.html#VirtualNode


# @functional_transform("virtual_node_mol")
class VirtualNodeMol(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
    (functional name: :obj:`virtual_node`).
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """

    def __call__(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        assert num_nodes is not None
        edge_type = data.get("edge_type", torch.zeros_like(row))

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)

        atom_feats = get_atom_feature_dims()
        bond_feats = get_bond_feature_dims()

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes,), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes,), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            if key == "edge_index" or key == "edge_type" or key == "y":
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == "edge_weight":
                    size[dim] = 2 * num_nodes
                    fill_value = 1.0
                elif key == "batch":
                    size[dim] = 1
                    fill_value = int(value[0])
                elif old_data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.0
                elif old_data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 0.0

                if (
                    fill_value is not None
                    and old_data.is_edge_attr(key)
                    and key != "bond_lengths"
                    and key != "bond_angles"
                ):
                    new_value = (
                        torch.tensor([d - 1 for d in bond_feats], dtype=torch.long)
                        .reshape(1, -1)
                        .repeat(2 * num_nodes, 1)
                    )
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)
                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if "num_nodes" in data:
            data.num_nodes = num_nodes + 1

        # Add the center of mass as the position of the virtual node
        masses = atomic_mass[data.x[:-1, 0]].reshape(-1, 1)
        c = (masses * data.positions[:-1]).sum(dim=0) / masses.sum()
        data.positions[-1] = c

        data.x[-1] = torch.tensor([d - 1 for d in atom_feats])

        return data


activation_dict = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU, "Swish": nn.SiLU}

"""
TorchDrug functions for variadic inputs, credit to https://torchdrug.ai/
"""


def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Example::

        >>> mask = multi_slice_mask(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([1, 1, 1, 0, 1, 1])).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    if slices.numel():
        assert slices.min() >= 0 and slices.max() <= length
    mask = torch_scatter.scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def variadic_to_padded(input: torch.Tensor, size: torch.LongTensor, value=0):
    """
    Convert a variadic tensor to a padded tensor.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        value (scalar): fill value for padding

    Returns:
        (Tensor, BoolTensor): padded tensor and mask
    """
    num_sample = len(size)
    max_size = size.max()
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    shape = (num_sample, max_size) + input.shape[1:]
    padded = torch.full(shape, value, dtype=input.dtype, device=size.device)
    padded[mask] = input
    return padded, mask


def padded_to_variadic(padded: torch.Tensor, size: torch.LongTensor):
    """
    Convert a padded tensor to a variadic tensor.

    Parameters:
        padded (Tensor): padded tensor of shape :math:`(N, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    num_sample, max_size = padded.shape[:2]
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    return padded[mask]
