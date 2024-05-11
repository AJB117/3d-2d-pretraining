import matplotlib.pyplot as plt
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


def distance_ranking_loss(batch, embs_3d, pred_head, sample_pairs, embs_2d=None):
    pass


def centrality_ranking_loss(batch, embs_3d, pred_head, sample_pairs, embs_2d=None):
    """
    Given a batch of embeddings, predict the centrality ranking of pairs of atoms
    """
    if embs_2d is not None:
        pair_embs_2d = embs_2d[sample_pairs[0]] + embs_2d[sample_pairs[1]]
        pair_embs_3d = embs_3d[sample_pairs[0]] + embs_3d[sample_pairs[1]]
        pair_embs = pair_embs_2d + pair_embs_3d
    else:
        pair_embs = embs_3d[sample_pairs[0]] + embs_3d[sample_pairs[1]]

    centralities = batch.eig_centrality
    centrality_a = centralities[sample_pairs[0]]
    centrality_b = centralities[sample_pairs[1]]

    diff = centrality_a - centrality_b
    diff[torch.abs(diff) < 1e-2] = 0

    # 2 classes: a is central, b is central
    is_a_central = torch.zeros_like(diff)
    is_a_central[diff < 0] = 1
    is_a_central[diff == 0] = 2

    pred_centralities = pred_head(pair_embs).squeeze()
    is_a_central = is_a_central

    acc = ((torch.argmax(pred_centralities, dim=1) == is_a_central).sum().item()) / len(
        is_a_central
    )

    loss = ce_loss(pred_centralities, is_a_central.long())

    return loss, acc


def betweenness_ranking_loss(batch, embs_3d, pred_head, sample_pairs, embs_2d=None):
    """
    Given a batch of embeddings, predict the centrality ranking of pairs of atoms
    """
    if embs_2d is not None:
        pair_embs_2d = embs_2d[sample_pairs[0]] + embs_2d[sample_pairs[1]]
        pair_embs_3d = embs_3d[sample_pairs[0]] + embs_3d[sample_pairs[1]]
        pair_embs = pair_embs_2d + pair_embs_3d
    else:
        pair_embs = embs_3d[sample_pairs[0]] + embs_3d[sample_pairs[1]]

    centralities = batch.betweenness_centrality

    centrality_a = centralities[sample_pairs[0]]
    centrality_b = centralities[sample_pairs[1]]

    diff = centrality_a - centrality_b
    diff[torch.abs(diff) < 1e-4] = 0

    # 3 classes: a is central, b is central, equal
    is_a_central = torch.zeros_like(diff)
    is_a_central[diff < 0] = 1
    is_a_central[diff == 0] = 2

    pred_centralities = pred_head(pair_embs).squeeze()

    acc = ((torch.argmax(pred_centralities, dim=1) == is_a_central).sum().item()) / len(
        is_a_central
    )

    loss = ce_loss(pred_centralities, is_a_central.long())

    return loss, acc


def spd_loss(batch, embs_3d, pred_head, sample_edges, embs_2d=None):
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

    spds = spds[flat_indices]

    if embs_2d is not None:
        pair_embs_2d = embs_2d[sample_edges[0]] + embs_2d[sample_edges[1]]
        pair_embs_3d = embs_3d[sample_edges[0]] + embs_3d[sample_edges[1]]
        pair_embs = pair_embs_2d + pair_embs_3d
    else:
        pair_embs = embs_3d[sample_edges[0]] + embs_3d[sample_edges[1]]

    pred_spds = pred_head(pair_embs).squeeze()

    acc = (torch.argmax(pred_spds, dim=1) == spds).sum().item() / len(spds)

    true_spds = spds
    loss = ce_loss(pred_spds, true_spds)

    return loss, acc


def anchor_pred_loss(embs, pred_head, indices):
    """
    Given a batch of embeddings, predict the anchor atom of each bond angle
    """

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


def interatomic_distance_loss(batch, embs_2d, pred_head, sample_edges, embs_3d=None):
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

    if embs_3d is not None:
        pair_embs_2d = embs_2d[sample_edges[0]] + embs_2d[sample_edges[1]]
        pair_embs_3d = embs_3d[sample_edges[0]] + embs_3d[sample_edges[1]]
        pair_embs = pair_embs_2d + pair_embs_3d
    else:
        pair_embs = embs_2d[sample_edges[0]] + embs_2d[sample_edges[1]]

    pred_distances = pred_head(pair_embs).squeeze()
    true_distances = interatomic_distances[sample_edges[0], sample_edges[1]]

    loss = mse_loss(pred_distances, true_distances)

    return loss


def bond_angle_loss(batch, embs_2d, pred_head, indices, embs_3d=None, rep_type="atom"):
    """
    Given a batch of embeddings, predict the bond angle between three atoms
    """
    angles = batch.bond_angles

    true_angles = angles[:, -1]
    anchor, src, dst = indices[:, 0], indices[:, 1], indices[:, 2]

    if rep_type == "atom":
        if embs_3d is not None:
            angle_emb_2d = torch.cat(
                [embs_2d[anchor], embs_2d[src], embs_2d[dst]],
                dim=1,
            ).to(embs_2d.device)
            angle_emb_3d = torch.cat(
                [embs_3d[anchor], embs_3d[src], embs_3d[dst]],
                dim=1,
            ).to(embs_3d.device)
            angle_emb = angle_emb_2d + angle_emb_3d
        else:
            angle_emb = torch.cat(
                [embs_2d[anchor], embs_2d[src], embs_2d[dst]],
                dim=1,
            ).to(embs_2d.device)
    elif rep_type == "bond":
        if embs_3d is not None:
            b1_emb_2d = embs_2d[src] + embs_2d[anchor]
            b2_emb_2d = embs_2d[dst] + embs_2d[anchor]

            b1_emb_3d = embs_3d[src] + embs_3d[anchor]
            b2_emb_3d = embs_3d[dst] + embs_3d[anchor]

            angle_emb_2d = torch.cat([b1_emb_2d, b2_emb_2d], dim=1).to(embs_2d.device)
            angle_emb_3d = torch.cat([b1_emb_3d, b2_emb_3d], dim=1).to(embs_3d.device)

            angle_emb = angle_emb_2d + angle_emb_3d
        else:
            b1_emb = embs_2d[src] + embs_2d[anchor]
            b2_emb = embs_2d[dst] + embs_2d[anchor]

            angle_emb = torch.cat([b1_emb, b2_emb], dim=1).to(embs_2d.device)

    pred_angles = pred_head(angle_emb).squeeze()
    loss = mse_loss(pred_angles, true_angles)

    return loss


def dihedral_angle_loss(
    batch,
    embs_2d,
    pred_head,
    indices,
    embs_3d=None,
    rep_type="atom",
    symm=False,
    visualize=False,
    as_classification=False,
):
    """
    Given a batch of embeddings, predict the dihedral angle between 4 atoms/2 bonds
    """
    dihedrals = batch.dihedral_angles

    if symm:
        # reflect all angles to be positive
        dihedrals[dihedrals < 0] = dihedrals[dihedrals < 0] * -1

    true_angles = torch.nan_to_num(dihedrals[:, -1])
    i, j, k, l = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

    if rep_type == "atom":
        if embs_3d is not None:
            dihedral_emb_2d = torch.cat(
                [
                    embs_2d[i],
                    embs_2d[j],
                    embs_2d[k],
                    embs_2d[l],
                ],
                dim=1,
            ).to(embs_2d.device)

            dihedral_emb_3d = torch.cat(
                [
                    embs_3d[i],
                    embs_3d[j],
                    embs_3d[k],
                    embs_3d[l],
                ],
                dim=1,
            ).to(embs_3d.device)
            dihedral_embs = dihedral_emb_2d + dihedral_emb_3d
        else:
            dihedral_embs = torch.cat(
                [
                    embs_2d[i],
                    embs_2d[j],
                    embs_2d[k],
                    embs_2d[l],
                ],
                dim=1,
            ).to(embs_2d.device)
    elif rep_type == "bond":
        if embs_3d is not None:
            b1_emb_2d = embs_2d[i] + embs_2d[j]
            b2_emb_2d = embs_2d[k] + embs_2d[j]
            b3_emb_2d = embs_2d[l] + embs_2d[k]

            b1_emb_3d = embs_3d[i] + embs_3d[j]
            b2_emb_3d = embs_3d[k] + embs_3d[j]
            b3_emb_3d = embs_3d[l] + embs_3d[k]

            dihedral_emb_2d = torch.cat([b1_emb_2d, b2_emb_2d, b3_emb_2d], dim=1).to(
                embs_2d.device
            )
            dihedral_emb_3d = torch.cat([b1_emb_3d, b2_emb_3d, b3_emb_3d], dim=1).to(
                embs_3d.device
            )

            dihedral_embs = dihedral_emb_2d + dihedral_emb_3d
        else:
            b1_emb = embs_2d[i] + embs_2d[j]
            b2_emb = embs_2d[k] + embs_2d[j]
            b3_emb = embs_2d[l] + embs_2d[k]

            dihedral_embs = torch.cat([b1_emb, b2_emb, b3_emb], dim=1).to(
                embs_2d.device
            )

    if as_classification:
        # bucket the angles into 10 classes
        true_angles = torch.floor(true_angles * 10).long()
        true_angles[true_angles == 10] = 9

        pred_angles = pred_head(dihedral_embs).squeeze()

        acc = (torch.argmax(pred_angles, dim=1) == true_angles).sum().item() / len(
            true_angles
        )
        loss = ce_loss(pred_angles, true_angles)
    else:
        pred_angles = pred_head(dihedral_embs).squeeze()
        acc = None
        loss = mse_loss(pred_angles, true_angles)

    if visualize:
        plt.hist(pred_angles.detach().cpu().numpy(), bins=50, color="red", alpha=0.5)
        plt.hist(true_angles.detach().cpu().numpy(), bins=50, color="blue", alpha=0.5)
        plt.show()

    return loss, acc


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


def edge_classification_loss(batch, embs_3d, pred_head, embs_2d=None):
    """
    Given a batch of embeddings, predict the bond type between two atoms
    """
    edges = batch.edge_index
    edge_attrs = batch.edge_attr

    if embs_2d is not None:
        pair_emb_2d = embs_2d[edges[0]] + embs_2d[edges[1]]
        pair_emb_3d = embs_3d[edges[0]] + embs_3d[edges[1]]

        edge_embs = pair_emb_2d + pair_emb_3d
    else:
        edge_embs = embs_3d[edges[0]] + embs_3d[edges[1]]

    pred = pred_head(edge_embs).squeeze()

    pos_labels = edge_attrs[:, 0]

    acc = (torch.argmax(pred, dim=1) == pos_labels).sum().item() / len(pos_labels)

    loss = ce_loss(pred, pos_labels)

    return loss, acc
