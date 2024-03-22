import wandb
import warnings
import csv
import pdb
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm

from config import args
from Geom3D.models import GNN, GraphPredLinear, SchNet, EGNN, GNN_NoAtom, Interactor
from Geom3D.datasets import (
    MoleculeDatasetQM9,
    GeneratedMoleculeDatasetQM9,
)
from splitters import QM9_random_customized_01, QM9_random_customized_02, QM9_50k_split
from util import VirtualNodeMol, apply_init

warnings.filterwarnings("ignore")


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).float()


def split(dataset, data_root):
    if args.split == "customized_01" and "QM9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = QM9_random_customized_01(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (01) on QM9")
    elif args.split == "customized_02" and "QM9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = QM9_random_customized_02(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (02) on QM9")
    elif args.split == "50k_split" and "QM9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = QM9_50k_split(
            dataset, null_value=0, seed=args.seed
        )
        print("50k split on QM9")
    else:
        raise ValueError("Invalid split option on {}.".format(args.dataset))
    print(len(train_dataset), "\t", len(valid_dataset), "\t", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset


def model_setup():
    if args.mode != "method":
        model = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=node_class,
        )
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)
        return model, graph_pred_linear

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
    )

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
            nn.Linear(intermediate_dim, num_tasks),
        )
    else:
        graph_pred_mlp = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_tasks),
        )

    for layer in graph_pred_mlp:
        if isinstance(layer, nn.Linear):
            apply_init(args.initialization)(layer.weight)

    return (
        model,
        graph_pred_mlp,
    )


def load_model(
    model_3d,
    model_2d,
    graph_pred_linear,
    model_weight_file,
):
    print("Loading from {}".format(model_weight_file))
    model_weight = torch.load(model_weight_file)
    if args.mode == "method":  # as opposed to baseline
        model_2d.load_state_dict(model_weight["model_2D"])

        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

        return

    if "model_3D" in model_weight:
        model_3d.load_state_dict(model_weight["model_3D"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    else:
        model_2d.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    return


def save_model(save_best):
    if args.output_model_dir == "":
        return

    if save_best:
        print(f"save model {model_name} with optimal loss")
        output_model_path = os.path.join(args.output_model_dir, f"{model_name}.pth")
    else:
        print(f"save model {model_name} in the last epoch")
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_final.pth"
        )

    saved_model_dict = {}

    saved_model_dict["model"] = model.state_dict()

    if graph_pred_mlp is not None:
        saved_model_dict["graph_pred_linear"] = graph_pred_mlp.state_dict()

    torch.save(saved_model_dict, output_model_path)

    return


def train(epoch, device, loader, optimizer):
    model.train()

    if graph_pred_mlp is not None:
        graph_pred_mlp.train()

    loss_acc = 0
    y_true, y_scores = [], []
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.mode != "method":
            molecule_rep = model(batch.x, batch.positions, batch=batch.batch)
            pred = graph_pred_mlp(molecule_rep).squeeze()
        else:
            interact_rep = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.positions, batch.batch
            )
            pred = graph_pred_mlp(interact_rep).squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]

        # normalize
        normalized_y = (y - TRAIN_mean) / TRAIN_std

        loss = criterion(pred, normalized_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        pred_denorm = (pred.detach() * TRAIN_std + TRAIN_mean) * loader.dataset.eV2meV[
            task_id
        ]
        mev_y = y * loader.dataset.eV2meV[task_id]

        y_true.append(mev_y)
        y_scores.append(pred_denorm)

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)

    return loss_acc, mae


@torch.no_grad()
def eval(device, loader):
    model.eval()

    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.mode != "method":
            molecule_rep = model(batch.x, batch.positions, batch=batch.batch)
            pred = graph_pred_mlp(molecule_rep).squeeze()
        else:
            interact_rep = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.positions, batch.batch
            )
            pred = graph_pred_mlp(interact_rep).squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        # denormalize
        pred = (pred * TRAIN_std + TRAIN_mean) * loader.dataset.eV2meV[task_id]
        y = y * loader.dataset.eV2meV[task_id]

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rotation_transform = None
    if args.use_rotation_transform:
        rotation_transform = RandomRotation()

    num_tasks = 1
    assert args.dataset == "QM9"

    transform = (
        VirtualNodeMol() if args.interaction_rep_3d in ("com", "const_radius") else None
    )
    data_root = "data/molecule_datasets/{}".format(args.dataset)
    dataset = MoleculeDatasetQM9(
        data_root,
        dataset=args.dataset,
        task=args.task,
        rotation_transform=rotation_transform,
        transform=transform,
        use_pure_atomic_num=False,
    )
    task_id = dataset.task_id

    if not args.use_generated_dataset:
        train_dataset, valid_dataset, test_dataset = split(dataset, data_root)
    else:
        _, valid_dataset, test_dataset = split(dataset, data_root)
        train_dataset = GeneratedMoleculeDatasetQM9(
            "data/generated_datasets/{}".format(args.dataset),
            dataset=args.dataset,
            task=args.task,
            rotation_transform=rotation_transform,
        )

    TRAIN_mean, TRAIN_std = (
        train_dataset.mean()[task_id].item(),
        train_dataset.std()[task_id].item(),
    )
    print("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class, edge_class = 120, 5
    (model, graph_pred_mlp) = model_setup()

    if args.input_model_file != "":
        load_model(
            model,
            graph_pred_mlp,
            args.input_model_file,
        )
    else:
        print("fine-tuning from scratch...")

    model = model.to(device)

    if graph_pred_mlp is not None:
        graph_pred_mlp.to(device)

    model_param_group = [{"params": model.parameters(), "lr": args.lr}]

    if graph_pred_mlp is not None:
        model_param_group.append({"params": graph_pred_mlp.parameters(), "lr": args.lr})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # set up optimizer
    # different learning rate for different part of GNN
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

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0

    args_dict = vars(args)

    if args.wandb:
        wandb.init(
            name=f"finetune_{args.dataset}_{args.output_model_name}",
            project="molecular-pretraining",
            config=args_dict,
        )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc, loss_mae = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_target, train_pred = eval(device, train_loader)
            else:
                train_mae = 0
            val_mae, val_target, val_pred = eval(device, val_loader)
            test_mae, test_target, test_pred = eval(device, test_loader)

            train_mae_list.append(loss_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            print(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    loss_mae, val_mae, test_mae
                )
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

                    filename = os.path.join(
                        args.output_model_dir, f"{model_name}_evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        val_target=val_target,
                        val_pred=val_pred,
                        test_target=test_target,
                        test_pred=test_pred,
                    )
        print("Took\t{}\n".format(time.time() - start_time))

    print(
        "{} best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            model_name,
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        )
    )

    with open("configs.csv", "a") as f:
        writer = csv.writer(f)
        loss_dict = {
            "finetune_train_mae": train_mae_list[best_val_idx],
            "finetune_val_mae": val_mae_list[best_val_idx],
            "finetune_test_mae": test_mae_list[best_val_idx],
        }

        dict_args = vars(args)
        dict_args.update(loss_dict)

        # just increment if no name is given

        dict_args["finetune_save_location"] = os.path.join(
            args.output_model_dir, f"{model_name}.pth"
        )

        writer.writerow(dict_args.values())

    save_model(save_best=False)
