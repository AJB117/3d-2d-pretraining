import pdb
import wandb
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import args
from Geom3D.datasets import MoleculeNetDataset2D
from Geom3D.models import GNN, Interactor
from splitters import scaffold_split
from util import get_num_task, apply_init

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


def split(dataset, smiles_list, data_root):
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset,
        smiles_list,
        null_value=0,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
    )
    print("split via scaffold")
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

    model = None

    if args.mode == "2d":
        model = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)
        print("# of params: ", sum(p.numel() for p in model.parameters()))
        return model, graph_pred_mlp

    model = Interactor(
        args,
        num_interaction_blocks=args.num_interaction_blocks,
        final_pool=args.final_pool,
        emb_dim=args.emb_dim,
        device=device,
        num_node_class=119 + 1,  # + 1 for virtual node
        interactor_type=args.interactor_type,
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

    print("# of params: ", sum(p.numel() for p in model.parameters()))
    return model, graph_pred_mlp


def load_model(model, model_weight_file):
    print("Loading from {}".format(model_weight_file))
    try:
        model_weights = torch.load(model_weight_file)
        if args.mode == "method":  # as opposed to baseline
            model.load_state_dict(model_weights["model"])
    except Exception as e:
        print("Failed to load model weights")
        print(e)
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
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.mode != "method":  # assumes 2d
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            mol_rep = global_mean_pool(node_repr, batch.batch)
        else:
            mol_rep = model.forward_2d(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

        # with torch.no_grad():
        #     mol_rep_3d = global_mean_pool(
        #         model.atom_encoder_3d(batch.x[:, 0]), batch.batch
        #     )

        if graph_pred_mlp is not None:
            # pred = graph_pred_mlp(torch.cat((mol_rep, mol_rep_3d), dim=-1))
            pred = graph_pred_mlp(mol_rep)
        else:
            pred = mol_rep

        y = batch.y.view(pred.shape).to(torch.float64)

        # PaiNN can have some nan values
        mol_is_nan = torch.isnan(pred)
        if torch.sum(mol_is_nan) > 0:
            continue

        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)

        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros(loss_mat.shape, device=device).to(torch.float64),
        )

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    if graph_pred_mlp is not None:
        graph_pred_mlp.eval()
    y_true, y_scores, y_valid = [], [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        # with torch.no_grad():
        #     mol_rep_3d = global_mean_pool(
        #         model.atom_encoder_3d(batch.x[:, 0]), batch.batch
        #     )

        if args.mode != "method":  # assumes 2d
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            mol_rep = global_mean_pool(node_repr, batch.batch)
        else:
            mol_rep = model.forward_2d(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

        if graph_pred_mlp is not None:
            # pred = graph_pred_mlp(torch.cat((mol_rep, mol_rep_3d), dim=-1))
            pred = graph_pred_mlp(mol_rep)
        else:
            pred = mol_rep

        true = batch.y.view(pred.shape)

        # PaiNN can have some nan values
        is_nan = torch.isnan(pred)
        # Whether y is non-null or not.
        is_valid = torch.logical_and((true**2 > 0), ~is_nan)

        y_true.append(true)
        y_scores.append(pred)
        y_valid.append(is_valid)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_valid = torch.cat(y_valid, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        try:
            is_valid = y_valid[:, i]
            roc_list.append(
                roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i])
            )
        except:
            print("{} is invalid".format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), y_true, y_scores


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

    num_tasks = get_num_task(args.dataset)
    data_root_2D = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = MoleculeNetDataset2D(data_root_2D, dataset=args.dataset)
    data_root = data_root_2D
    smiles_file = "{}/processed/smiles.csv".format(data_root)
    smiles_list = pd.read_csv(smiles_file, header=None)[0].tolist()

    train_dataset, valid_dataset, test_dataset = split(dataset, smiles_list, data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class, edge_class = 119, 4
    model, graph_pred_mlp = model_setup()

    if args.input_model_file != "":
        load_model(model, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_mlp is not None:
        graph_pred_mlp.to(device)
    print(graph_pred_mlp)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
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

    train_roc_list, val_roc_list, test_roc_list = [], [], []
    best_val_roc, best_val_idx = 0, 0

    args_dict = vars(args)

    if args.wandb:
        wandb.init(
            name=f"finetune_{args.dataset}_{args.output_model_name}",
            project="molecular-pretraining",
            config=args_dict,
        )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_roc, train_target, train_pred = eval(device, train_loader)
            else:
                train_roc = 0
            val_roc, val_target, val_pred = eval(device, val_loader)
            test_roc, test_target, test_pred = eval(device, test_loader)

            train_roc_list.append(train_roc)
            val_roc_list.append(val_roc)
            test_roc_list.append(test_roc)
            print(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    train_roc, val_roc, test_roc
                )
            )

            if args.wandb:
                wandb.log(
                    {
                        f"{args.dataset}_train_roc": train_roc,
                        f"{args.dataset}_val_roc": val_roc,
                        f"{args.dataset}_test_roc": test_roc,
                    }
                )

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = len(train_roc_list) - 1
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
            train_roc_list[best_val_idx] * 100,
            val_roc_list[best_val_idx] * 100,
            test_roc_list[best_val_idx] * 100,
        )
    )

    if args.wandb:
        wandb.log(
            {
                f"{args.dataset}_finetune_train_roc": train_roc_list[best_val_idx],
                f"{args.dataset}_finetune_val_roc": val_roc_list[best_val_idx],
                f"{args.dataset}_finetune_test_roc": test_roc_list[best_val_idx],
            }
        )
        wandb.finish()

    save_model(save_best=False)
