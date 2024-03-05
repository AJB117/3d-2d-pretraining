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

from config import args
from Geom3D.models import GNN, EGNN
from Geom3D.datasets import PCQM4Mv2
from ogb.lsc import PCQM4Mv2Evaluator


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
    if args.model_3d == "EGNN":
        model = EGNN(
            in_node_nf=args.emb_dim_egnn,
            in_edge_nf=args.emb_dim_egnn,
            hidden_nf=args.emb_dim_egnn,
            n_layers=args.n_layers_egnn,
            positions_weight=args.positions_weight_egnn,
            attention=args.attention_egnn,
            node_attr=False,
        )
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)
        return model, graph_pred_linear

    if args.model_2d == "GIN":
        model = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        )
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)
    else:
        raise Exception("2D model {} not included.".format(args.model_2d))
    return model, graph_pred_linear


def load_model(model, graph_pred_linear, model_weight_file, mode="MoleculeSDE"):
    print("Loading from {}".format(model_weight_file))
    model_weight = torch.load(model_weight_file)

    if mode != "MoleculeSDE":
        model.load_state_dict(model_weight["model"])
        return

    if "model_2D" in model_weight:
        model.load_state_dict(model_weight["model_2D"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    return


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.model_3d == "EGNN":
            node_repr, pos_3D_repr = model(
                batch.x, batch.positions, batch.edge_index, batch.edge_attr
            )
            molecule_repr = global_mean_pool(node_repr, batch.batch)
        elif args.model_2d == "GIN":
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            molecule_repr = global_mean_pool(node_repr, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_repr).squeeze()
        else:
            pred = molecule_repr.squeeze()

        y = batch.y

        # normalize
        y = (y - mean) / std

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.model_3d == "EGNN":
            node_repr, pos_3D_repr = model(
                batch.x, batch.positions, batch.edge_index, batch.edge_attr
            )
            molecule_repr = global_mean_pool(node_repr, batch.batch)
        elif args.model_2d == "GIN":
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            molecule_repr = global_mean_pool(node_repr, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_repr).squeeze()
        else:
            pred = molecule_repr.squeeze()

        y = batch.y

        # denormalize pred
        pred = pred * std + mean

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
        mean, std = torch.load(
            os.path.join(args.output_model_dir, "pcqm4mv2_mean_std.pth")
        )
    except FileNotFoundError:
        mean, std = (
            dataset.mean(),
            dataset.std(),
        )
        torch.save(
            (mean, std),
            os.path.join(args.output_model_dir, "pcqm4mv2_mean_std.pth"),
        )
    
    print("mean: {:.6f}\tstd: {:.6f}".format(mean, std))

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
    model, graph_pred_linear = model_setup()

    if args.input_model_file != "":
        print("loading model from {}".format(args.input_model_file))
        load_model(model, graph_pred_linear, args.input_model_file, args.mode)
    else:
        print("fine-tuning from scratch...")

    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    print(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
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

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=25,
        min_lr=1e-6,
        cooldown=20,
        threshold=1e-4,
        verbose=True,
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_target, train_pred = eval(device, train_loader)
            else:
                train_mae = 0
            val_mae, val_target, val_pred = eval(device, val_loader)
            test_mae, test_target, test_pred = eval(device, test_loader)

            scheduler.step(val_mae)

            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            print(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    train_mae, val_mae, test_mae
                )
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
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
        "best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        )
    )

    save_model(save_best=False)
