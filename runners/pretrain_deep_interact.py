import wandb
import sys
import uuid
import csv
import pdb
import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import batched_negative_sampling

from Geom3D.datasets import Molecule3DDataset, MoleculeDatasetQM9
from Geom3D.models import GNN, SchNet, EGNN, Interactor
from config import args
from ogb.utils.features import get_atom_feature_dims
from util import NTXentLoss, perturb, VirtualNodeMol
from collections import defaultdict


ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()


def compute_accuracy(pred, target):
    return float(
        torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()
    ) / len(pred)


def save_model(
    save_best,
    graph_pred_mlp,
    model,
    model_name,
    pretrained_heads_2d=None,
    pretrained_heads_3d=None,
):
    if args.output_model_dir == "":
        return

    saver_dict = {
        "graph_pred_mlp": graph_pred_mlp.state_dict(),
        "model": model.state_dict(),
        "pretrained_heads_2d": [head.state_dict() for head in pretrained_heads_2d],
        "pretrained_heads_3d": [head.state_dict() for head in pretrained_heads_3d],
    }
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}\n".format(optimal_loss))
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_complete.pth"
        )
    else:
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_complete_final.pth"
        )

    torch.save(saver_dict, output_model_path)


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
    return (true_positives / num_positives + true_negatives / num_negatives) / 2


def create_pretrain_heads(task, intermediate_dim, device):
    normalizer = None
    if args.batch_norm:
        normalizer = nn.BatchNorm1d(intermediate_dim)
    elif args.layer_norm:
        normalizer = nn.LayerNorm(intermediate_dim)
    else:
        normalizer = nn.Identity()

    if task == "interatomic_dist":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
    elif task == "edge_existence":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )

    return pred_head.to(device)


def interatomic_distance_loss(batch, embs, pred_head, max_samples=10):
    """
    Given a batch of embeddings, predict the interatomic distances
    with the given head and return the mean squared error loss.
    """
    interatomic_distances = batch.atomic_distances

    num_nodes_per_batch = torch.bincount(batch.batch).to(embs.device)
    num_samples_per_batch = torch.min(
        torch.tensor(max_samples), num_nodes_per_batch
    ).to(embs.device)

    pairs = torch.cat(
        [
            torch.randint(0, num_nodes_per_batch[i], (num_samples_per_batch[i], 2)).to(
                embs.device
            )
            + torch.sum(num_nodes_per_batch[:i])
            for i in range(len(num_nodes_per_batch))
        ],
        dim=0,
    ).to(embs.device)

    pair_embs = torch.cat([embs[pairs[:, 0]], embs[pairs[:, 1]]], dim=1).to(embs.device)

    pred_distances = pred_head(pair_embs).squeeze()
    pdb.set_trace()  # bug: batches could go beyond the 50 atom limit; maybe just compute the distances at runtime
    true_distances = interatomic_distances[pairs[:, 0], pairs[:, 1]]

    loss = mse_loss(pred_distances, true_distances)

    return loss


def edge_existence_loss(batch, embs, pred_head, max_samples=10):
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
        batch.edge_index, batch.batch, num_neg_samples=max_samples
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


def pretrain(
    args,
    model_name,
    model: Interactor,
    device,
    loader,
    optimizer,
    graph_pred_mlp=None,
    lr_scheduler=None,
    epoch=0,
    pretrain_heads_2d=None,
    pretrain_heads_3d=None,
):
    loss_dict = defaultdict(int)
    start_time = time.time()

    CL_loss_accum, CL_acc_accum = 0, 0
    loss_accum = 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader

    num_iters = len(loader)

    for step, batch in enumerate(l):
        batch = batch.to(device)

        final_embs, midstream_2d_outs, midstream_3d_outs = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.positions,
            batch.batch,
            require_midstream=True,
        )

        tasks_2d = args.pretrain_2d_tasks
        tasks_3d = args.pretrain_3d_tasks

        loss_terms = []
        loss = 0

        if args.start_tasks_from_end:
            pretrain_heads_2d = pretrain_heads_2d[::-1]
            pretrain_heads_3d = pretrain_heads_3d[::-1]
            midstream_2d_outs = midstream_2d_outs[::-1]
            midstream_3d_outs = midstream_3d_outs[::-1]

        for task_2d, pred_head, midstream in zip(
            tasks_2d, pretrain_heads_2d, midstream_2d_outs
        ):
            if task_2d == "interatomic_dist":
                new_loss = interatomic_distance_loss(batch, midstream, pred_head)
                loss += new_loss

            loss_terms.append(new_loss)
            loss_dict[task_2d] += new_loss.item()

        for task_3d, pred_head, midstream in zip(
            tasks_3d, pretrain_heads_3d, midstream_3d_outs
        ):
            if task_3d == "edge_existence":
                new_loss = edge_existence_loss(batch, midstream, pred_head)
                loss += new_loss

            loss_terms.append(new_loss)
            loss_dict[task_3d] += new_loss.item()

        loss_dict["loss_accum"] += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)
            loss_dict["loss_accum"] /= len(loader)

    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_dict["loss_accum"])

    global optimal_loss

    for key in loss_dict:
        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"] and key == "loss_accum":
            continue

        loss_dict[key] /= len(loader)

    print("Total Loss: {:.5f}".format(loss_dict["loss_accum"]))

    print("Time: {:.5f}\n".format(time.time() - start_time))

    temp_loss = loss_dict["loss_accum"]

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(
            save_best=True,
            graph_pred_mlp=graph_pred_mlp,
            model=model,
            model_name=model_name,
            pretrained_heads_2d=pretrain_heads_2d,
            pretrained_heads_3d=pretrain_heads_3d,
        )

    loss_dict["loss_accum"] = loss_accum.item()

    if args.wandb:
        wandb.log(loss_dict)

    return loss_dict


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    num_node_classes = 119 + 1  # add 1 for virtual node

    transform = (
        VirtualNodeMol() if args.interaction_rep_3d in ("com", "const_radius") else None
    )
    data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    try:
        dataset = Molecule3DDataset(
            data_root,
            args.dataset,
            mask_ratio=args.SSL_masking_ratio,
            remove_center=True,
            use_extend_graph=args.use_extend_graph,
            transform=transform,
        )
    except FileNotFoundError:
        if args.dataset == "QM9":
            data_root = "data/molecule_datasets/{}".format(args.dataset)
            use_pure_atomic_num = False
            if args.model_3d == "EGNN":
                use_pure_atomic_num = False
            MoleculeDatasetQM9(
                data_root,
                dataset=args.dataset,
                task=args.task,
                use_pure_atomic_num=use_pure_atomic_num,
            )
        dataset = Molecule3DDataset(
            data_root,
            args.dataset,
            mask_ratio=args.SSL_masking_ratio,
            remove_center=True,
            use_extend_graph=args.use_extend_graph,
            transform=transform,
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # set up models
    model = Interactor(
        args,
        num_interaction_blocks=args.num_interaction_blocks,
        final_pool=args.final_pool,
        emb_dim=args.emb_dim,
        device=device,
        num_node_class=119 + 1,  # + 1 for virtual node
        interaction_rep_2d=args.interaction_rep_2d,
        interaction_rep_3d=args.interaction_rep_3d,
        residual=args.residual,
        model_2d=args.model_2d,
        model_3d=args.model_3d,
        dropout=args.dropout_ratio,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
    ).to(device)

    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    normalizer = None
    if args.batch_norm:
        normalizer = nn.BatchNorm1d(intermediate_dim)
    elif args.layer_norm:
        normalizer = nn.LayerNorm(intermediate_dim)
    else:
        normalizer = nn.Identity()

    if args.final_pool == "cat":
        graph_pred_mlp = nn.Sequential(
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        ).to(device)
    else:
        graph_pred_mlp = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        ).to(device)

    pretrain_heads_2d = [
        create_pretrain_heads(task, intermediate_dim, device)
        for task in args.pretrain_2d_tasks
    ]
    pretrain_heads_3d = [
        create_pretrain_heads(task, intermediate_dim, device)
        for task in args.pretrain_3d_tasks
    ]

    for layer in graph_pred_mlp:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    if args.input_model_file != "":
        model_weight = torch.load(args.input_model_file)
        model.load_state_dict(model_weight["model"])
        print("successfully loaded model checkpoint")

    model_param_group = []

    # non-GNN components
    model_param_group.append({"params": graph_pred_mlp.parameters(), "lr": args.lr})
    for head in pretrain_heads_2d:
        model_param_group.append({"params": head.parameters(), "lr": args.lr})
    for head in pretrain_heads_3d:
        model_param_group.append({"params": head.parameters(), "lr": args.lr})

    # GNNs
    model_param_group.append(
        {"params": model.blocks_2d.parameters(), "lr": args.lr * args.gnn_2d_lr_scale}
    )
    model_param_group.append(
        {"params": model.blocks_3d.parameters(), "lr": args.lr * args.gnn_3d_lr_scale}
    )

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
        print("Apply lr scheduler ReduceLROnPlateau")
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))

    model_name = args.output_model_name
    # just use date if we stick with the default
    if model_name == "":
        now_in_ms = int(time.time() * 1000)
        model_name = f"{now_in_ms}"

    args_dict = vars(args)

    if args.wandb:
        wandb.init(project="molecular-pretraining", config=args_dict)

    for epoch in range(1, args.epochs + 1):
        print("epoch: {}".format(epoch))
        loss_dict = pretrain(
            args,
            model_name,
            model,
            device,
            loader,
            optimizer,
            graph_pred_mlp=graph_pred_mlp,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            pretrain_heads_2d=pretrain_heads_2d,
            pretrain_heads_3d=pretrain_heads_3d,
        )

    # to aggregate later with a separate script
    config_fname = f"{args.process_num}_config.csv"
    with open(config_fname, "w") as f:
        pass

    with open(config_fname, "r") as f:
        reader = csv.reader(f)
        num_rows = len(list(reader))

    config_id = str(uuid.uuid4())

    with open(config_fname, "a") as f:
        writer = csv.writer(f)

        args_dict = vars(args)
        args_dict.update(loss_dict)

        args_dict["pretrain_save_location"] = os.path.join(
            args.output_model_dir, f"{model_name}_complete.pth"
        )

        current_git_hash = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )

        # to be updated in the downstream task training runs
        args_dict["finetune_train_mae"] = 0
        args_dict["finetune_val_mae"] = 0
        args_dict["finetune_test_mae"] = 0
        args_dict["finetune_save_location"] = ""
        args_dict["git_hash"] = current_git_hash
        args_dict["cmd"] = " ".join(sys.argv)
        args_dict["id"] = config_id

        header = args_dict.keys()
        if num_rows == 0:
            writer.writerow(header)

        writer.writerow(args_dict.values())

    save_model(
        save_best=False,
        graph_pred_mlp=graph_pred_mlp,
        model=model,
        model_name=model_name,
        pretrained_heads_2d=pretrain_heads_2d,
        pretrained_heads_3d=pretrain_heads_3d,
    )

    if args.wandb:
        wandb.finish()

    return config_id


if __name__ == "__main__":
    main()
