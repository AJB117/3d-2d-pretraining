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

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from Geom3D.datasets import Molecule3DDataset, MoleculeDatasetQM9
from Geom3D.models import GNN, SchNet, EGNN
from config import args
from ogb.utils.features import get_atom_feature_dims
from util import NTXentLoss, perturb, CL_acc


ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


def compute_accuracy(pred, target):
    return float(
        torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()
    ) / len(pred)


def save_model(
    save_best,
    graph_pred_linear,
    down_project,
    model_2D,
    model_2D_pos,
    model_3D,
    enforcer_mlp,
    model_name,
    optimal_loss,
    process_num,
):
    if args.output_model_dir == "":
        return

    saver_dict = {
        "graph_pred_linear": graph_pred_linear.state_dict(),
        "down_project": down_project.state_dict(),
        "model_2D": model_2D.state_dict(),
        "model_2D_pos": model_2D_pos.state_dict(),
        "model_3D": model_3D.state_dict(),
        "enforcer_mlp": enforcer_mlp.state_dict(),
    }
    if save_best:
        print("save model with loss: {:.5f}\n".format(optimal_loss))
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_{process_num}_complete.pth"
        )
    else:
        output_model_path = os.path.join(
            args.output_model_dir, f"{model_name}_{process_num}_complete_final.pth"
        )

    torch.save(saver_dict, output_model_path)


def pool(x, batch, pool="mean"):
    if pool == "mean":
        return global_mean_pool(x, batch)
    elif pool == "sum":
        return global_add_pool(x, batch)
    elif pool == "max":
        return global_max_pool(x, batch)
    else:
        raise NotImplementedError("Pool type not included.")


def train(
    args,
    model_name,
    model_2D,
    model_2D_pos,
    model_3D,
    device,
    loader,
    optimizer,
    balances=[
        1,
        1,
        1,
        1,
    ],  # hyperparams corresponding to invariant loss, equivariant loss, hybrid loss, matching loss
    graph_pred_linear=None,
    down_project=None,
    enforcer_mlp=None,
    bond_length_predictor=None,
    bond_angle_predictor=None,
    lr_scheduler=None,
    epoch=0,
    optimal_loss=1e10,
):
    start_time = time.time()

    model_2D.train()
    model_2D_pos.train()
    model_3D.eval()

    CL_loss_accum, CL_acc_accum = 0, 0
    CL_tp_rate_accum, CL_tn_rate_accum = 0, 0
    rot_loss_accum = 0
    trans_loss_accum = 0
    bond_length_loss_accum = 0
    bond_angle_loss_accum = 0
    matching_loss_accum = 0
    loss_accum = 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader

    num_iters = len(loader)

    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_2D_repr = model_2D(batch.x, batch.edge_index, batch.edge_attr)
        node_2D_pos_repr = model_2D_pos(
            batch.x, batch.edge_index, batch.edge_attr
        )  # for synthetic coords

        # invariant loss
        mol_2d_repr = pool(node_2D_repr, batch.batch, pool=args.graph_pooling)
        mol_2d_repr = graph_pred_linear(mol_2d_repr)

        if args.model_3d == "EGNN":
            node_3D_repr, _ = model_3D(
                batch.x, batch.positions, batch.edge_index, batch.edge_attr
            )
            mol_3d_repr = pool(node_3D_repr, batch.batch, pool=args.graph_pooling)
        elif args.model_3d == "SchNet":
            mol_3d_repr = model_3D(batch.x, batch.positions, batch.batch)

        invariant_loss = balances[0] * NTXentLoss(mol_2d_repr, mol_3d_repr)
        CL_loss_accum += invariant_loss
        CL_acc_accum += CL_acc(mol_2d_repr, mol_3d_repr)[0]
        CL_tp_rate_accum += CL_acc(mol_2d_repr, mol_3d_repr)[1]
        CL_tn_rate_accum += CL_acc(mol_2d_repr, mol_3d_repr)[2]

        # equivariant loss
        pos_synth = down_project(node_2D_pos_repr)
        pos_enforced = enforcer_mlp(pos_synth)

        rotated_pos_synth, translated_pos_synth, _, _ = perturb(
            pos_synth, device=device
        )
        rotated_pos_enforced, translated_pos_enforced, _, _ = perturb(
            batch.positions, device=device
        )

        rot_loss = (
            torch.norm(
                (pos_enforced @ rotated_pos_enforced) - (pos_synth @ rotated_pos_synth),
                p="fro",
                dim=1,
            )
            .pow(2)
            .mean()
        )
        rot_loss_accum += rot_loss

        trans_loss = (
            torch.norm(translated_pos_enforced - pos_enforced, p="fro", dim=1)
            .pow(2)
            .mean()
            + torch.norm(translated_pos_synth - pos_synth, p="fro", dim=1).pow(2).mean()
        )
        trans_loss_accum += trans_loss

        equiv_loss = balances[1] * (rot_loss + trans_loss)

        # hybrid loss
        node_2D_repr = torch.cat([node_2D_repr, pos_synth], dim=1)  # N x (F + 3)

        ## bond length prediction
        # bond_lengths = batch.bond_lengths  # E x 1, [edge_idx, length]
        i_emb = node_2D_repr[batch.edge_index[0]]
        j_emb = node_2D_repr[batch.edge_index[1]]

        dist_vecs = (
            batch.positions[batch.edge_index[0]] - batch.positions[batch.edge_index[1]]
        )
        bond_lengths = dist_vecs.norm(dim=1).unsqueeze(1)

        bond_length_pred = bond_length_predictor(torch.cat([i_emb, j_emb], dim=1))
        bond_length_loss = mse_loss(bond_length_pred, bond_lengths)
        bond_length_loss_accum += bond_length_loss

        ## bond angle prediction
        bond_angles = batch.bond_angles  # E' x 3 x 1, [anchor, src, dst, angle]
        anchor_emb = node_2D_repr[bond_angles[:, 0].long()]
        src_emb = node_2D_repr[bond_angles[:, 1].long()]
        dst_emb = node_2D_repr[bond_angles[:, 2].long()]
        bond_angle_targets = bond_angles[:, 3].unsqueeze(1)

        bond_angle_pred = bond_angle_predictor(
            torch.cat([src_emb, anchor_emb, dst_emb], dim=1)
        )
        bond_angle_loss = mse_loss(bond_angle_pred, bond_angle_targets)
        bond_angle_loss_accum += bond_angle_loss

        hybrid_loss = balances[2] * (bond_length_loss + bond_angle_loss)

        # matching loss
        matching_loss = balances[3] * mse_loss(pos_synth, batch.positions)
        matching_loss_accum += matching_loss

        loss = invariant_loss + equiv_loss + hybrid_loss + matching_loss
        loss_accum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_accum)

    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    CL_tp_rate_accum /= len(loader)
    CL_tn_rate_accum /= len(loader)
    rot_loss_accum /= len(loader)
    trans_loss_accum /= len(loader)
    bond_length_loss_accum /= len(loader)
    bond_angle_loss_accum /= len(loader)
    matching_loss_accum /= len(loader)
    loss_accum /= len(loader)

    print("CL Loss: {:.5f}\tCL Acc: {:.5f}".format(CL_loss_accum, CL_acc_accum))
    print(
        "CL TP Rate: {:.5f}\tCL TN Rate: {:.5f}".format(
            CL_tp_rate_accum, CL_tn_rate_accum
        )
    )
    print(
        "Rot Loss: {:.5f}\tTrans Loss: {:.5f}".format(rot_loss_accum, trans_loss_accum)
    )
    print(
        "Bond Length Loss: {:.5f}\tBond Angle Loss: {:.5f}".format(
            bond_length_loss_accum, bond_angle_loss_accum
        )
    )
    print("Matching Loss: {:.5f}".format(matching_loss_accum))
    print("Total Loss: {:.5f}".format(loss_accum))

    print("Time: {:.5f}\n".format(time.time() - start_time))

    temp_loss = loss_accum

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(
            save_best=True,
            graph_pred_linear=graph_pred_linear,
            down_project=down_project,
            model_2D=model_2D,
            model_2D_pos=model_2D_pos,
            model_3D=model_3D,
            enforcer_mlp=enforcer_mlp,
            model_name=model_name,
            optimal_loss=optimal_loss,
            process_num=args.process_num,
        )

    loss_dict = {
        "CL_loss_accum": CL_loss_accum.item(),
        "CL_acc_accum": CL_acc_accum.item(),
        "rot_loss_accum": rot_loss_accum.item(),
        "trans_loss_accum": trans_loss_accum.item(),
        "bond_length_loss_accum": bond_length_loss_accum.item(),
        "bond_angle_loss_accum": bond_angle_loss_accum.item(),
        "matching_loss_accum": matching_loss_accum.item(),
        "pretrain_loss_accum": loss_accum.item(),
        "pretrain_optimal_loss": optimal_loss.item(),
    }

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

    num_node_classes = 119
    num_edge_classes = 3

    transform = None
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
            use_pure_atomic_num = True
            if args.model_3d == "EGNN":
                use_pure_atomic_num = False
            MoleculeDatasetQM9(
                data_root,
                dataset=args.dataset,
                task=args.task,
                use_pure_atomic_num=use_pure_atomic_num,
            ).process()
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
    molecule_model_2D = GNN(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)

    molecule_model_2D_pos = GNN(
        args.num_layer_pos,
        args.emb_dim_pos,
        JK=args.JK_pos,
        drop_ratio=args.dropout_ratio_pos,
        gnn_type=args.gnn_type_pos,
    ).to(device)

    print("Using 3d model\t", args.model_3d)
    if args.model_3d == "SchNet":
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=num_node_classes,
        ).to(device)
    elif args.model_3d == "EGNN":
        molecule_model_3D = EGNN(
            in_node_nf=args.emb_dim_egnn,
            in_edge_nf=args.emb_dim_egnn,
            hidden_nf=args.emb_dim_egnn,
            n_layers=args.n_layers_egnn,
            positions_weight=args.positions_weight_egnn,
            attention=args.attention_egnn,
            node_attr=False,
        ).to(device)

    else:
        raise NotImplementedError("Model {} not included.".format(args.model_3d))

    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    graph_pred_linear = nn.Sequential(
        nn.Linear(intermediate_dim, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, intermediate_dim),
    ).to(device)

    down_project = nn.Sequential(  # down project the pos GNN to the coordinate space
        nn.Linear(args.emb_dim_pos, args.emb_dim_pos // 2),
        nn.BatchNorm1d(args.emb_dim_pos // 2),
        nn.ReLU(),
        nn.Linear(
            args.emb_dim_pos // 2,
            args.emb_dim_pos // 2,
        ),
        nn.BatchNorm1d(args.emb_dim_pos // 2),
        nn.ReLU(),
        nn.Linear(
            args.emb_dim_pos // 2,
            args.emb_dim_pos // 4,
        ),
        nn.Linear(
            args.emb_dim_pos // 4,
            3,
        ),
    ).to(device)

    enforcer_mlp = nn.Sequential(  # enforce equivariance
        nn.Linear(3, 3),
        nn.BatchNorm1d(3),
        nn.ReLU(),
        nn.Linear(3, 3),
    ).to(device)

    bond_length_predictor = nn.Sequential(
        nn.Linear((intermediate_dim + 3) * 2, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, 1),
    ).to(device)

    bond_angle_predictor = nn.Sequential(
        nn.Linear((intermediate_dim + 3) * 3, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, 1),
    ).to(device)

    if args.input_model_file_3d != "":
        model_weight = torch.load(args.input_model_file_3d)
        molecule_model_3D.load_state_dict(model_weight["model"])
        print("successfully loaded 3D model")

    model_param_group = []

    # MLP heads
    model_param_group.append({"params": graph_pred_linear.parameters(), "lr": args.lr})
    model_param_group.append({"params": down_project.parameters(), "lr": args.lr})
    model_param_group.append({"params": enforcer_mlp.parameters(), "lr": args.lr})
    model_param_group.append(
        {"params": bond_length_predictor.parameters(), "lr": args.lr}
    )
    model_param_group.append(
        {"params": bond_angle_predictor.parameters(), "lr": args.lr}
    )

    # GNNs
    model_param_group.append(
        {"params": molecule_model_2D.parameters(), "lr": args.lr * args.gnn_2d_lr_scale}
    )
    model_param_group.append(
        {
            "params": molecule_model_2D_pos.parameters(),
            "lr": args.lr * args.gnn_2d_pos_lr_scale,
        }
    )
    model_param_group.append(
        {"params": molecule_model_3D.parameters(), "lr": args.lr * args.gnn_3d_lr_scale}
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

    for epoch in range(1, args.epochs + 1):
        print("epoch: {}".format(epoch))
        loss_dict, optimal_loss = train(
            args,
            model_name,
            molecule_model_2D,
            molecule_model_2D_pos,
            molecule_model_3D,
            device,
            loader,
            optimizer,
            graph_pred_linear=graph_pred_linear,
            down_project=down_project,
            enforcer_mlp=enforcer_mlp,
            bond_length_predictor=bond_length_predictor,
            bond_angle_predictor=bond_angle_predictor,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
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

        dict_args = vars(args)
        dict_args.update(loss_dict)

        dict_args["pretrain_save_location"] = os.path.join(
            args.output_model_dir, f"{model_name}_{args.process_num}_complete.pth"
        )

        # to be updated in the downstream task training runs
        dict_args["finetune_train_mae"] = 0
        dict_args["finetune_val_mae"] = 0
        dict_args["finetune_test_mae"] = 0
        dict_args["finetune_save_location"] = ""
        dict_args["id"] = config_id

        header = dict_args.keys()
        if num_rows == 0:
            writer.writerow(header)

        writer.writerow(dict_args.values())

    save_model(
        save_best=False,
        graph_pred_linear=graph_pred_linear,
        down_project=down_project,
        model_2D=molecule_model_2D,
        model_2D_pos=molecule_model_2D_pos,
        model_3D=molecule_model_3D,
        enforcer_mlp=enforcer_mlp,
        model_name=model_name,
        optimal_loss=optimal_loss,
        process_num=args.process_num,
    )

    return config_id


if __name__ == "__main__":
    main()
