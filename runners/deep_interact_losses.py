import fun
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import batched_negative_sampling, to_dense_adj
from scipy.sparse.csgraph import floyd_warshall

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()


def get_global_atom_indices(indices, batch, num_per_batch):
    """
    Adjust the atom indices to be global indices
    indices is of shape (N, K) where N is the number of samples and K is the number of atoms
    """

    sizes = torch.bincount(batch).to(indices.device)
    node_offset = torch.cat(
        [torch.tensor([0]).to(indices.device), torch.cumsum(sizes[:-1], dim=0)]
    )

    global_offsets = (
        node_offset.repeat_interleave(num_per_batch)
        .unsqueeze(1)
        .repeat(1, indices.shape[1])
    )

    global_indices = indices + global_offsets

    return global_indices.long()


def get_batched_flattened_indices(flattened_matrix, edge_index, batch):
    """
    Retrieve the values of the flattened matrix at the given edge indices
    """
    device = flattened_matrix.device
    sizes = torch.bincount(batch)

    batch_u = batch[edge_index[0]]

    node_offset = torch.cat(
        [torch.tensor([0]).to(device), torch.cumsum(sizes[:-1], dim=0)]
    )
    local_idx_u = edge_index[0] - node_offset[batch[edge_index[0]]]
    local_idx_v = edge_index[1] - node_offset[batch[edge_index[1]]]

    graph_offset = torch.cat(
        [torch.tensor([0]).to(device), torch.cumsum(sizes[:-1] ** 2, dim=0)]
    )

    flat_indices = graph_offset[batch_u] + sizes[batch_u] * local_idx_u + local_idx_v

    return flat_indices


def get_sample_edges(batch, num_samples):
    """
    Generate a list of edges to sample from each batch
    """
    num_nodes_per_batch = (
        torch.bincount(batch.batch).to(batch.edge_index.device).cpu().numpy()
    )
    sample_edges = (
        torch.tensor(fun.generate_pairs(num_nodes_per_batch, num_samples))
        .long()
        .to(batch.edge_index.device)
    )

    return sample_edges


def centrality_ranking_loss(batch, embs, pred_head, sample_pairs):
    """
    Given a batch of embeddings, predict the centrality ranking of pairs of atoms
    """
    pair_embs = embs[sample_pairs[0]] + embs[sample_pairs[1]]
    centralities = batch.eig_centrality
    centrality_a = centralities[sample_pairs[0]]
    centrality_b = centralities[sample_pairs[1]]

    is_a_central = (centrality_a > centrality_b).long()
    pred_centralities = pred_head(pair_embs).squeeze()

    loss = bce_loss(pred_centralities, is_a_central.float())

    return loss


def spd_loss(batch, embs, pred_head, sample_edges):
    """
    Given a batch of embeddings, predict the shortest path distance between 2 atoms
    """
    spds = batch.spd_mat  # linearized spd matrix

    try:
        assert spds[spds == 0].numel() == batch.x.shape[0]
    except Exception as e:
        # print(e)
        # pdb.set_trace()
        return None

    flat_indices = get_batched_flattened_indices(spds, sample_edges, batch.batch)
    # if step == 41:
    #     import pdb
    #     pdb.set_trace()

    spds = spds[flat_indices]

    pair_embs = embs[sample_edges[0]] + embs[sample_edges[1]]

    pred_spds = pred_head(pair_embs).squeeze()
    true_spds = spds
    loss = ce_loss(pred_spds, true_spds)

    return loss


def anchor_pred_loss(embs, pred_head, indices):
    """
    Given a batch of embeddings, predict the anchor atom of each bond angle
    """
    # angles = batch.bond_angles
    # indices = angles[:, :-1]

    permuted_indices = torch.rand_like(indices.float()).argsort(dim=1)
    anchor_indices = (permuted_indices == 0).nonzero()[:, 1]

    a_embs = embs[permuted_indices[:, 0]]
    b_embs = embs[permuted_indices[:, 1]]
    c_embs = embs[permuted_indices[:, 2]]

    angle_embs = torch.cat([a_embs, b_embs, c_embs], dim=1).to(embs.device)

    pred_anchors = pred_head(angle_embs).squeeze()
    loss = ce_loss(pred_anchors, anchor_indices)

    return loss


def anchor_tup_pred_loss(embs, pred_head, indices):
    """
    Given a batch of embeddings, predict the two anchor atoms of each dihedral angle
    """

    # dihedrals = batch.dihedral_angles
    # indices = dihedrals[:, :-1]

    permuted_indices = torch.rand_like(indices.float()).argsort(dim=1)
    anchor_indices_1 = (permuted_indices == 0).nonzero()[:, 1]
    anchor_indices_2 = (permuted_indices == 1).nonzero()[:, 1]

    a_embs = embs[permuted_indices[:, 0]]
    b_embs = embs[permuted_indices[:, 1]]
    c_embs = embs[permuted_indices[:, 2]]
    d_embs = embs[permuted_indices[:, 3]]

    dihedral_embs = torch.cat([a_embs, b_embs, c_embs, d_embs], dim=1).to(embs.device)

    pred_anchors = pred_head(dihedral_embs).squeeze()
    anchor_indices_1_oh = F.one_hot(anchor_indices_1, num_classes=4).float()
    anchor_indices_2_oh = F.one_hot(anchor_indices_2, num_classes=4).float()

    target_anchors = anchor_indices_1_oh + anchor_indices_2_oh
    loss = bce_loss(pred_anchors, target_anchors)

    return loss


def interatomic_distance_loss(batch, embs, pred_head, sample_edges):
    """
    Given a batch of embeddings, predict the interatomic distances
    with the given head and return the mean squared error loss.
    """
    positions = batch.positions
    interatomic_diffs = positions.unsqueeze(0) - positions.unsqueeze(1)
    interatomic_distances = torch.norm(interatomic_diffs, dim=-1)
    max_num_nodes = batch.num_nodes

    interatomic_distances = F.pad(
        interatomic_distances, (0, max_num_nodes - interatomic_distances.size(0))
    )
    pair_embs = embs[sample_edges[0]] + embs[sample_edges[1]]

    pred_distances = pred_head(pair_embs).squeeze()
    true_distances = interatomic_distances[sample_edges[0], sample_edges[1]]

    loss = mse_loss(pred_distances, true_distances)

    return loss


def bond_angle_loss(batch, embs, pred_head, indices):
    """
    Given a batch of embeddings, predict the bond angle between three atoms
    """
    angles = batch.bond_angles

    true_angles = angles[:, -1]

    # b1_emb = embs[indices[:, 1]] + embs[indices[:, 0]]
    # b2_emb = embs[indices[:, 2]] + embs[indices[:, 0]]

    # angle_embs = torch.cat([b1_emb, b2_emb], dim=1).to(embs.device)
    angle_emb = torch.cat(
        [embs[indices[:, 0]], embs[indices[:, 1]], embs[indices[:, 2]]], dim=1
    ).to(embs.device)

    pred_angles = pred_head(angle_emb).squeeze()
    loss = mse_loss(pred_angles, true_angles)

    return loss


def dihedral_angle_loss(batch, embs, pred_head, indices):
    """
    Given a batch of embeddings, predict the dihedral angle between 4 atoms/2 bonds
    """
    dihedrals = batch.dihedral_angles

    true_angles = torch.nan_to_num(dihedrals[:, -1])

    # b1_emb = embs[indices[:, 0]] + embs[indices[:, 1]]
    # b2_emb = embs[indices[:, 2]] + embs[indices[:, 1]]
    # b3_emb = embs[indices[:, 3]] + embs[indices[:, 2]]

    dihedral_embs = torch.cat(
        [
            # b1_emb,
            # b2_emb,
            # b3_emb,
            embs[indices[:, 0]],
            embs[indices[:, 1]],
            embs[indices[:, 2]],
            embs[indices[:, 3]],
        ],
        dim=1,
    ).to(embs.device)

    pred_angles = pred_head(dihedral_embs).squeeze()
    loss = mse_loss(pred_angles, true_angles)

    return loss


def edge_existence_loss(batch, embs, pred_head, neg_samples=50):
    """
    Given a batch of embeddings, predict whether an edge exists between
    two atoms with the given head and return the binary cross entropy loss.
    Predict the existence of all edges in the batch and sample negative
    edges for balance.
    """
    pos_links = batch.edge_index
    pos_link_embs = embs[pos_links[0]] + embs[pos_links[1]]

    neg_links = batched_negative_sampling(
        batch.edge_index, batch.batch, num_neg_samples=neg_samples
    ).to(embs.device)
    neg_link_embs = embs[neg_links[0]] + embs[neg_links[1]]

    pred_pos_links = pred_head(pos_link_embs).squeeze()
    pred_neg_links = pred_head(neg_link_embs).squeeze()

    pos_labels = torch.ones(len(pos_links[0])).to(embs.device)
    neg_labels = torch.zeros(len(neg_links[0])).to(embs.device)

    pred_edges = torch.cat([pred_pos_links, pred_neg_links], dim=0)
    true_edges = torch.cat([pos_labels, neg_labels], dim=0)

    loss = bce_loss(pred_edges, true_edges)

    return loss


def edge_classification_loss(batch, embs, pred_head):
    """
    Given a batch of embeddings, predict the bond type between two atoms
    """
    edges = batch.edge_index
    edge_attrs = batch.edge_attr
    edge_embs = embs[edges[0]] + embs[edges[1]]

    pred = pred_head(edge_embs).squeeze()

    pos_labels = edge_attrs[:, 0]

    loss = ce_loss(pred, pos_labels)

    return loss
