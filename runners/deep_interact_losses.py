import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import batched_negative_sampling

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()


def interatomic_distance_loss(batch, embs, pred_head, max_samples=10):
    """
    Given a batch of embeddings, predict the interatomic distances
    with the given head and return the mean squared error loss.
    """
    positions = batch.positions
    interatomic_diffs = positions.unsqueeze(0) - positions.unsqueeze(1)
    interatomic_distances = torch.norm(interatomic_diffs, dim=-1)
    num_nodes_per_batch = torch.bincount(batch.batch).to(embs.device)
    max_num_nodes = batch.num_nodes

    interatomic_distances = F.pad(
        interatomic_distances, (0, max_num_nodes - interatomic_distances.size(0))
    )

    pairs = []
    for i in range(len(num_nodes_per_batch)):
        offset = torch.sum(num_nodes_per_batch[:i])
        if max_samples > 0:
            combinations = torch.randint(0, num_nodes_per_batch[i], (max_samples, 2))
        else:
            combinations = torch.combinations(torch.arange(num_nodes_per_batch[i]))
        combinations = torch.cat([combinations, combinations[:, [1, 0]]], dim=0).to(
            embs.device
        )
        pairs.append(combinations + offset)

    pairs = torch.cat(pairs, dim=0)
    pair_embs = torch.cat([embs[pairs[:, 0]], embs[pairs[:, 1]]], dim=1).to(embs.device)

    pred_distances = pred_head(pair_embs).squeeze()
    true_distances = interatomic_distances[pairs[:, 0], pairs[:, 1]]

    loss = mse_loss(pred_distances, true_distances)

    return loss


def edge_existence_loss(batch, embs, pred_head, neg_samples=20):
    """
    Given a batch of embeddings, predict whether an edge exists between
    two atoms with the given head and return the binary cross entropy loss.
    Predict the existence of all edges in the batch and sample negative
    edges for balance.
    """
    pos_links = batch.edge_index
    pos_link_embs = torch.cat([embs[pos_links[0]], embs[pos_links[1]]], dim=1).to(
        embs.device
    )

    neg_links = batched_negative_sampling(
        batch.edge_index, batch.batch, num_neg_samples=neg_samples
    ).to(embs.device)
    neg_link_embs = torch.cat([embs[neg_links[0]], embs[neg_links[1]]], dim=1).to(
        embs.device
    )

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
    pos_links = batch.edge_index
    pos_link_attrs = batch.edge_attr
    pos_link_embs = torch.cat([embs[pos_links[0]], embs[pos_links[1]]], dim=1).to(
        embs.device
    )

    pred_pos_links = pred_head(pos_link_embs).squeeze()

    pos_labels = pos_link_attrs[:, 0]

    loss = bce_loss(pred_pos_links, pos_labels)

    return loss
