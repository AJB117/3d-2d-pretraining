from torchviz import make_dot
import wandb
import sys
import uuid
import csv
import pdb
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import warnings

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from torch_geometric.nn import global_mean_pool
from Geom3D.datasets import Molecule3DDataset, MoleculeDatasetQM9, PCQM4Mv2
from Geom3D.models import Interactor
from config import args
from ogb.utils.features import get_atom_feature_dims
from util import VirtualNodeMol, NTXentLoss
from collections import defaultdict
from deep_interact_losses import (
    edge_classification_loss,
    edge_existence_loss,
    spd_loss,
    interatomic_distance_loss,
    bond_angle_loss,
    dihedral_angle_loss,
    anchor_pred_loss,
    anchor_tup_pred_loss,
    get_sample_edges,
    get_global_atom_indices,
    centrality_ranking_loss,
)

# warnings.filterwarnings("ignore")


def split_pretraining(dataset, dataset_name):
    split_idx = dataset.get_idx_split()  # only use the train/valid/test-dev splits for fine-tuning, otherwise use the train set

    try:
        pretrain_idx = torch.load(f"{dataset_name}_pretraining_idx.pt")
    except FileNotFoundError:
        train_idx = split_idx["train"]
        full_len = len(train_idx)
        pretrain_idx = list(
            filter(
                lambda x: x < full_len - 567,
                train_idx,
            )
        )
        torch.save(pretrain_idx, f"{dataset_name}_pretraining_idx.pt")

    train_dataset = dataset[pretrain_idx]

    print("train set for pretraining: ", len(train_dataset))
    return train_dataset


def split(dataset):
    split_idx = dataset.get_idx_split()  # only use the train/valid/test-dev splits for fine-tuning, otherwise use the train set

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["valid"]

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    print(len(train_dataset), "\t", len(valid_dataset), "\t", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset


# From GPT-4
def pretty_print_dict(d):
    if not d:
        print("The dictionary is empty.")
        return

    max_key_length = max(len(str(key)) for key in d.keys())
    max_value_length = max(len(str(value)) for value in d.values())

    print(f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)}")
    print("-" * (max_key_length + max_value_length + 3))

    for key, value in d.items():
        print(
            f"{str(key).ljust(max_key_length)} | {str(round(value, 5)).ljust(max_value_length)}"
        )


ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
n_bond_types = 4


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
    optimal_loss=1e10,
    epoch=0,
):
    if args.output_model_dir == "":
        return

    saver_dict = {
        "graph_pred_mlp": graph_pred_mlp.state_dict(),
        "model": model.state_dict(),
        "pretrained_heads_2d": [head.state_dict() for head in pretrained_heads_2d],
        "pretrained_heads_3d": [head.state_dict() for head in pretrained_heads_3d],
        "epoch": epoch,
    }
    if save_best:
        print("save model with total loss: {:.5f}\n".format(optimal_loss))
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_complete.pth"
        )
    else:
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_complete_final.pth"
        )

    print(f"Saving model to {output_model_path}")

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
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
    elif task == "edge_existence":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
    elif task == "bond_angle":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 3, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
    elif task == "dihedral_angle":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 4, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
    elif task == "edge_classification":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, n_bond_types),
        )
    elif task == "spd":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 64),  # max number of shortest path distances
        )
    elif task == "bond_anchor_pred":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 3, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 3),
        )
    elif task == "dihedral_anchor_pred":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim * 4, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 4),
        )
    elif task == "centrality_ranking":
        pred_head = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, 3),
        )

    return pred_head.to(device)


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
    optimal_loss=1e10,
):
    loss_dict = defaultdict(int)
    start_time = time.time()

    loss_accum = 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader

    num_iters = len(loader)

    for step, batch in enumerate(l):
        if step == 10:
            break
        batch = batch.to(device)

        final_embs, midstream_2d_outs, midstream_3d_outs = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.positions,
            batch.batch,
            require_midstream=True,
        )

        pretrain_2d_task_indices = args.pretrain_2d_task_indices
        pretrain_3d_task_indices = args.pretrain_3d_task_indices

        if args.pretrain_strategy == "geometric":
            tasks_2d = args.pretrain_2d_tasks
            tasks_3d = args.pretrain_3d_tasks
            balances_2d = args.pretrain_2d_balances
            balances_3d = args.pretrain_3d_balances

            loss_terms = []
            loss = 0

            outs_2d = [midstream_2d_outs[i] for i in pretrain_2d_task_indices]
            outs_3d = [midstream_3d_outs[i] for i in pretrain_3d_task_indices]

            if args.cl_per_block:
                for i, (out_2d, out_3d) in enumerate(zip(outs_2d, outs_3d)):
                    out_2d_mol = global_mean_pool(out_2d, batch.batch)
                    out_3d_mol = global_mean_pool(out_3d, batch.batch)

                    loss = loss + NTXentLoss(out_2d_mol, out_3d_mol)
                    loss_terms.append(loss)
                    loss_dict["CL_block_{}".format(i)] += loss.item()

            if "interatomic_dist" in tasks_2d or "spd" in tasks_3d:
                sample_edges = get_sample_edges(
                    batch, args.pretrain_interatomic_samples
                ).T

            bond_angle_indices = get_global_atom_indices(
                batch.bond_angles[:, :-1], batch.batch, batch.num_angles
            )
            dihedral_angle_indices = get_global_atom_indices(
                batch.dihedral_angles[:, :-1], batch.batch, batch.num_dihedrals
            )

            if args.all_losses_at_end:
                final_midstream_2d = midstream_2d_outs[-1]
                final_midstream_3d = midstream_3d_outs[-1]

                for i, (task_2d, task_3d) in enumerate(zip(tasks_2d, tasks_3d)):
                    if task_2d == "interatomic_dist":
                        loss_2d = interatomic_distance_loss(
                            batch,
                            final_midstream_2d,
                            pretrain_heads_2d[i],
                            sample_edges,
                        )
                    elif task_2d == "bond_angle":
                        loss_2d = bond_angle_loss(
                            batch,
                            final_midstream_2d,
                            pretrain_heads_2d[i],
                            bond_angle_indices,
                        )
                    elif task_2d == "dihedral_angle":
                        loss_2d = dihedral_angle_loss(
                            batch,
                            final_midstream_2d,
                            pretrain_heads_2d[i],
                            dihedral_angle_indices,
                        )

                    if task_3d == "edge_existence":
                        loss_3d = edge_existence_loss(
                            batch,
                            final_midstream_3d,
                            pretrain_heads_3d[i],
                            neg_samples=args.pretrain_neg_link_samples,
                        )
                    elif task_3d == "edge_classification":
                        loss_3d = edge_classification_loss(
                            batch, final_midstream_3d, pretrain_heads_3d[i]
                        )
                    elif task_3d == "spd":
                        loss_3d = spd_loss(
                            batch,
                            final_midstream_3d,
                            pretrain_heads_3d[i],
                            sample_edges,
                        )
                    elif task_3d == "bond_anchor_pred":
                        loss_3d = anchor_pred_loss(
                            final_midstream_3d,
                            pretrain_heads_3d[i],
                            bond_angle_indices,
                        )
                    elif task_3d == "dihedral_anchor_pred":
                        loss_3d = anchor_tup_pred_loss(
                            final_midstream_3d,
                            pretrain_heads_3d[i],
                            dihedral_angle_indices,
                        )
                    elif task_3d == "centrality_ranking":
                        loss_3d = centrality_ranking_loss(
                            batch, final_midstream_3d, pretrain_heads_3d[i], sample_edges
                        )

                    loss = loss + loss_2d + loss_3d
                    loss_terms.append(loss)
                    loss_dict[task_2d] += loss_2d.item()
                    loss_dict[task_3d] += loss_3d.item()
            else:
                for task_2d, pred_head, midstream, balance_2d in zip(
                    tasks_2d, pretrain_heads_2d, outs_2d, balances_2d
                ):
                    if task_2d == "interatomic_dist":
                        new_loss = interatomic_distance_loss(
                            batch,
                            midstream,
                            pred_head,
                            sample_edges,
                        )
                    elif task_2d == "bond_angle":
                        new_loss = bond_angle_loss(
                            batch, midstream, pred_head, bond_angle_indices
                        )
                    elif task_2d == "dihedral_angle":
                        new_loss = dihedral_angle_loss(
                            batch, midstream, pred_head, dihedral_angle_indices
                        )

                    new_loss = balance_2d * new_loss
                    loss = loss + new_loss
                    loss_terms.append(new_loss)
                    loss_dict[task_2d] += new_loss.item()

                for task_3d, pred_head, midstream, balance_3d in zip(
                    tasks_3d, pretrain_heads_3d, outs_3d, balances_3d
                ):
                    if task_3d == "edge_existence":
                        new_loss = edge_existence_loss(
                            batch,
                            midstream,
                            pred_head,
                            neg_samples=args.pretrain_neg_link_samples,
                        )
                    elif task_3d == "edge_classification":
                        new_loss = edge_classification_loss(batch, midstream, pred_head)
                    elif task_3d == "spd":
                        new_loss = spd_loss(batch, midstream, pred_head, sample_edges)
                    elif task_3d == "bond_anchor_pred":
                        new_loss = anchor_pred_loss(
                            midstream, pred_head, bond_angle_indices
                        )
                    elif task_3d == "dihedral_anchor_pred":
                        new_loss = anchor_tup_pred_loss(
                            midstream, pred_head, dihedral_angle_indices
                        )
                    elif task_3d == "centrality_ranking":
                        new_loss = centrality_ranking_loss(
                            batch, midstream, pred_head, sample_edges
                        )

                    new_loss = balance_3d * new_loss
                    loss = loss + new_loss
                    loss_terms.append(new_loss)
                    loss_dict[task_3d] += new_loss.item()

        elif args.pretraining_strategy == "masking":
            pass  # ! TODO: Implement masking strategy

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

    for key in loss_dict:
        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"] and key == "loss_accum":
            continue

        loss_dict[key] /= len(loader)

    print("Total Loss: {:.5f}".format(loss_dict["loss_accum"]))

    print("Time: {:.5f}\n".format(time.time() - start_time))

    loss_dict["loss_accum"] = loss_dict["loss_accum"].item()

    pretty_print_dict(loss_dict)
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
            optimal_loss=optimal_loss,
            epoch=epoch,
        )

    loss_dict["optimal_loss"] = optimal_loss

    if args.wandb:
        wandb.log(loss_dict)

    return loss_dict, optimal_loss


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
    if args.dataset.startswith("QM9"):
        # data_root = "data/molecule_datasets/{}".format(args.dataset)
        data_root = f"{args.input_data_dir}/{args.dataset}"
        use_pure_atomic_num = False
        if args.model_3d == "EGNN":
            use_pure_atomic_num = False

        dataset = MoleculeDatasetQM9(
            data_root,
            dataset=args.dataset,
            task=args.task,
            use_pure_atomic_num=use_pure_atomic_num,
        )

        # dataset = Molecule3DDataset(
        #     data_root,
        #     args.dataset,
        #     mask_ratio=args.SSL_masking_ratio,
        #     remove_center=True,
        #     use_extend_graph=args.use_extend_graph,
        #     transform=transform,
        # )
    elif args.dataset == "PCQM4Mv2":
        base_dataset = PCQM4Mv2(data_root, transform=None)
        dataset, _, _ = split(base_dataset)
    elif args.dataset == "PCQM4Mv2-pretraining":
        dataset = PCQM4Mv2(data_root, transform=None)
        # dataset = split_pretraining(base_dataset)
    elif args.dataset == "PCQM4Mv2-pretraining-centrality":
        dataset = PCQM4Mv2(data_root, transform=None)
        # dataset = split_pretraining(base_dataset)
    elif args.dataset == "PCQM4Mv2-full-pretraining":
        base_dataset = PCQM4Mv2(data_root, transform=None)
        dataset = split_pretraining(base_dataset, args.dataset)

    print("# data points: ", len(dataset))

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

    epoch_start = 1

    if args.input_model_file != "":
        saver_dict = torch.load(args.input_model_file)
        model.load_state_dict(saver_dict["model"])
        graph_pred_mlp.load_state_dict(saver_dict["graph_pred_mlp"])
        for head, pretrained_head in zip(
            pretrain_heads_2d, saver_dict["pretrained_heads_2d"]
        ):
            head.load_state_dict(pretrained_head)
        for head, pretrained_head in zip(
            pretrain_heads_3d, saver_dict["pretrained_heads_3d"]
        ):
            head.load_state_dict(pretrained_head)
        epoch_start = saver_dict["epoch"]

        print("successfully loaded model checkpoint to resume pretraining with")

    model_param_group = []

    # non-GNN components
    model_param_group.append({"params": graph_pred_mlp.parameters(), "lr": args.lr})
    for head in pretrain_heads_2d:
        model_param_group.append({"params": head.parameters(), "lr": args.lr})
    for head in pretrain_heads_3d:
        model_param_group.append({"params": head.parameters(), "lr": args.lr})

    # model
    model_param_group.append(
        {"params": model.parameters(), "lr": args.lr * args.gnn_2d_lr_scale}
    )
    assert args.gnn_2d_lr_scale == args.gnn_3d_lr_scale

    print("# parameters: ", sum(p.numel() for p in model.parameters()))

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
        wandb.init(
            name=f"pretrain_{args.dataset}_{args.output_model_name}",
            project="molecular-pretraining",
            config=args_dict,
        )

    for epoch in range(epoch_start, args.epochs + 1):
        print("epoch: {}".format(epoch))
        loss_dict, optimal_loss = pretrain(
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
            optimal_loss=optimal_loss,
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
        optimal_loss=optimal_loss,
        epoch=epoch,
    )

    if args.wandb:
        wandb.finish()

    return config_id


if __name__ == "__main__":
    main()
