import csv
import os
import time

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import wandb

from config import args
from Geom3D.models import GNN, GraphPredLinear, SchNet, EGNN, GNN_NoAtom, Interactor
from Geom3D.datasets import PCQM4Mv2
from ogb.lsc import PCQM4Mv2Evaluator
from util import apply_init


class PCQM4MEvaluatorWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = PCQM4Mv2Evaluator()
        self.val_only = False

    def forward(self, preds, targets):
        if preds.shape[1] != 1:
            return torch.tensor(float("NaN"))
        # input_dict = {"y_true": targets.long().squeeze(), "y_pred": preds.squeeze()}
        input_dict = {"y_true": targets.squeeze(), "y_pred": preds.squeeze()}
        return torch.tensor(self.evaluator.eval(input_dict)["mae"])


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def load_model(model, model_weight_file, model_3d=None, model_2d=None):
    print("Loading from {}".format(model_weight_file))
    model_weights = torch.load(model_weight_file, map_location=device)
    if args.mode == "method":  # as opposed to baseline
        model.load_state_dict(model_weights["model"])
    return


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


def model_setup():
    normalizer = None
    if args.batch_norm:
        normalizer = nn.BatchNorm1d(intermediate_dim)
    elif args.layer_norm:
        normalizer = nn.LayerNorm(intermediate_dim)
    else:
        normalizer = nn.Identity()

    if args.use_3d_only or args.use_2d_only:
        graph_pred_mlp = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_tasks),
        )
    elif args.mode == "method" and args.final_pool == "cat":
        graph_pred_mlp = nn.Sequential(
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            normalizer,
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_tasks),
        )

    model_2d, model_3d = None, None

    if args.mode == "3d":
        model_3d = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=119 + 1,
        )
    elif args.mode == "2d":
        model_2d = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)

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

    for layer in graph_pred_mlp:
        if isinstance(layer, nn.Linear):
            apply_init(args.initialization)(layer.weight)

    return model, graph_pred_mlp, model_2d, model_3d


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
    num_iters = len(loader)

    y_true, y_scores = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.mode != "method":  # assumes 3d
            mol_rep = model(batch.x, batch.positions, batch=batch.batch)
        else:
            if args.use_3d_only:
                mol_rep = model.forward_3d(batch.x, batch.positions, batch.batch)
            elif args.use_2d_only:
                mol_rep = model.forward_2d(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
            else:
                mol_rep = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.positions,
                    batch.batch,
                )

            pred = graph_pred_mlp(mol_rep).squeeze()

        y = batch.y

        # normalize
        y = (y - TRAIN_mean) / TRAIN_std

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        pred_denorm = pred.detach() * TRAIN_std + TRAIN_mean

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

        y_true.append(y)
        y_scores.append(pred_denorm)

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
    if graph_pred_mlp is not None:
        graph_pred_mlp.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.mode != "method":  # assumes 3d
            mol_rep = model(batch.x, batch.positions, batch=batch.batch)
        else:
            if args.use_3d_only:
                mol_rep = model.forward_3d(batch.x, batch.positions, batch.batch)
            elif args.use_2d_only:
                mol_rep = model.forward_2d(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
            else:
                mol_rep = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.positions,
                    batch.batch,
                )

            pred = graph_pred_mlp(mol_rep).squeeze()

        if graph_pred_mlp is not None:
            pred = graph_pred_mlp(mol_rep).squeeze()
        else:
            pred = mol_rep.squeeze()

        y = batch.y

        # denormalize pred
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


def get_random_indices(length, seed):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices


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

    num_tasks = 1

    assert args.dataset == "PCQM4Mv2"
    data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = PCQM4Mv2(data_root, transform=None)

    if args.require_3d:
        train_dataset, _, _ = split(dataset)

        total = len(train_dataset)
        val_len = int(total * 0.05)
        test_len = int(total * 0.05)
        train_len = total - val_len - test_len

        all_idx = get_random_indices(total, args.seed)
        train_idx = all_idx[:train_len]
        val_idx = all_idx[train_len : train_len + val_len]
        test_idx = all_idx[train_len + val_len :]

        train_dataset_3d = train_dataset[train_idx]
        valid_dataset_2d = train_dataset[val_idx]
        test_dataset_2d = train_dataset[test_idx]
    else:
        train_dataset_3d, valid_dataset_2d, test_dataset_2d = split(dataset)

    try:
        TRAIN_mean, TRAIN_std = torch.load(
            os.path.join(args.output_model_dir, "pcqm4mv2_mean_std.pth")
        )
    except FileNotFoundError:
        TRAIN_mean, TRAIN_std = (
            dataset.mean(),
            dataset.std(),
        )
        torch.save(
            (TRAIN_mean, TRAIN_std),
            os.path.join(args.output_model_dir, "pcqm4mv2_mean_std.pth"),
        )

    print("mean: {:.6f}\tstd: {:.6f}".format(TRAIN_mean, TRAIN_std))

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}
    train_loader = DataLoaderClass(
        train_dataset_3d,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )
    val_loader = DataLoaderClass(
        valid_dataset_2d,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )
    test_loader = DataLoaderClass(
        test_dataset_2d,
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

    num_node_classes, num_edge_classes = 119, 5
    model, graph_pred_mlp, model_2d, model_3d = model_setup()

    if args.input_model_file != "":
        try:
            load_model(
                model, args.input_model_file, model_3d=model_3d, model_2d=model_2d
            )
        except Exception as e:
            print(e)
            print(
                "Failed to load model from {}; fine-tuning from scratch".format(
                    args.input_model_file
                )
            )
    else:
        print("fine-tuning from scratch...")

    model.to(device)

    print(model)

    if graph_pred_mlp is not None:
        graph_pred_mlp.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    print("# of params: ", sum(p.numel() for p in model.parameters()))

    if graph_pred_mlp is not None:
        model_param_group.append({"params": graph_pred_mlp.parameters(), "lr": args.lr})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

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
        print("Epoch for {}: {}\nLoss: {}".format(epoch, "gap", loss_acc))

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

            if args.wandb:
                wandb.log(
                    {
                        "gap_train_mae": loss_mae,
                        "gap_val_mae": val_mae,
                        "gap_test_mae": test_mae,
                    }
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
            "gap_finetune_train_mae": train_mae_list[best_val_idx],
            "gap_finetune_val_mae": val_mae_list[best_val_idx],
            "gap_finetune_test_mae": test_mae_list[best_val_idx],
        }

        dict_args = vars(args)
        dict_args.update(loss_dict)

        # just increment if no name is given

        dict_args["finetune_save_location"] = os.path.join(
            args.output_model_dir, f"{model_name}.pth"
        )

        writer.writerow(dict_args.values())

    if args.wandb:
        wandb.log(
            {
                "gap_finetune_train_mae": train_mae_list[best_val_idx],
                "gap_finetune_val_mae": val_mae_list[best_val_idx],
                "gap_finetune_test_mae": test_mae_list[best_val_idx],
            }
        )

        wandb.finish()

    save_model(save_best=False)
