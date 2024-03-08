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


def perturb(
    pos,
    rotation=None,
    translation=None,
    rot_mu=0,
    rot_sigma=0.1,
    trans_mu=0,
    trans_sigma=1,
    device="cuda",
):
    if rotation is None:
        random_matrix = torch.normal(
            rot_mu, rot_sigma, size=(pos.shape[1], pos.shape[1])
        )
        rotation, _ = torch.qr(random_matrix)

    if translation is None:
        translation = torch.normal(trans_mu, trans_sigma, size=(1, pos.shape[1]))

    rotation = rotation.to(device)
    translation = translation.to(device)

    rotated_pos = rotation @ pos.T
    translated_pos = pos + translation

    return (
        rotated_pos,
        translated_pos,
        rotation,
        translation,
    )  # return rotation and translation for debugging


def save_model(save_best):
    if args.output_model_dir == "":
        return

    saver_dict = {
        "graph_pred_linear": graph_pred_linear.state_dict(),
        "down_project": down_project.state_dict(),
        "model_2D": molecule_model_2D.state_dict(),
        "model_2D_pos": molecule_model_2D.state_dict(),
        "model_3D": molecule_model_3D.state_dict(),
    }
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        output_model_path = os.path.join(args.output_model_dir, "model_complete.pth")
    else:
        output_model_path = os.path.join(
            args.output_model_dir, "model_complete_final.pth"
        )

    torch.save(saver_dict, output_model_path)


# credit to 3D Infomax (https://github.com/HannesStark/3DInfomax) for the following code
def NTXentLoss(z1: torch.Tensor, z2: torch.Tensor, tau=0.1, norm=True):
    sim_matrix = torch.einsum("ik, jk->ij", z1, z2)

    if norm:
        z1 = z1.norm(dim=1)
        z2 = z2.norm(dim=1)
        sim_matrix = sim_matrix / (torch.einsum("i, j->ij", z1, z2) + 1e-8)

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix.diag()

    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()

    return loss


def pool(x, batch, pool="mean"):
    if pool == "mean":
        return global_mean_pool(x, batch)
    elif pool == "sum":
        return global_add_pool(x, batch)
    elif pool == "max":
        return global_max_pool(x, batch)
    else:
        raise NotImplementedError("Pool type not included.")


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


def train(
    args,
    molecule_model_2D,
    molecule_model_2D_pos,
    molecule_model_3D,
    device,
    loader,
    optimizer,
    balances=[
        1,
        1,
        1,
        1,
    ],  # hyperparams corresponding to invariant loss, equivariant loss, hybrid loss, matching loss
):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_2D_pos.train()
    molecule_model_3D.eval()

    CL_loss_accum, CL_acc_accum = 0, 0
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
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_2D_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        node_2D_pos_repr = molecule_model_2D_pos(
            batch.x, batch.edge_index, batch.edge_attr
        )  # for synthetic coords

        if args.model_3d == "EGNN":
            node_3D_repr, pos_3D_repr = molecule_model_3D(
                batch.x, batch.positions, batch.edge_index, batch.edge_attr
            )

        # invariant loss
        mol_2d_repr = pool(node_2D_repr, batch.batch, pool=args.graph_pooling)
        mol_2d_repr = graph_pred_linear(mol_2d_repr)

        mol_3d_repr = pool(node_3D_repr, batch.batch, pool=args.graph_pooling)

        invariant_loss = balances[0] * NTXentLoss(mol_2d_repr, mol_3d_repr)
        CL_loss_accum += invariant_loss
        CL_acc_accum += CL_acc(mol_2d_repr, mol_3d_repr)

        # equivariant loss
        pos_synth = down_project(node_2D_pos_repr)

        rotated_pos, translated_pos, _, _ = perturb(node_2D_pos_repr, device=device)
        rotated_pos_synth, translated_pos_synth, _, _ = perturb(
            pos_synth, device=device
        )

        rot_loss = (
            node_2D_pos_repr @ rotated_pos - pos_synth @ rotated_pos_synth
        ).mean()
        rot_loss_accum += rot_loss

        trans_loss = (translated_pos - node_2D_pos_repr).mean() + (
            translated_pos_synth - pos_synth
        ).mean()
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
        matching_loss = balances[3] * mse_loss(pos_synth, pos_3D_repr)
        matching_loss_accum += matching_loss

        loss = invariant_loss + equiv_loss + hybrid_loss + matching_loss
        loss_accum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)

    rot_loss_accum /= len(loader)
    trans_loss_accum /= len(loader)
    bond_length_loss_accum /= len(loader)
    bond_angle_loss_accum /= len(loader)
    matching_loss_accum /= len(loader)
    loss_accum /= len(loader)

    temp_loss = 0  # todo

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print("CL Loss: {:.5f}\tCL Acc: {:.5f}".format(CL_loss_accum, CL_acc_accum))
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
    return


if __name__ == "__main__":
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

    down_project = nn.Sequential(
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

    for epoch in range(1, args.epochs + 1):
        print("epoch: {}".format(epoch))
        train(
            args,
            molecule_model_2D,
            molecule_model_2D_pos,
            molecule_model_3D,
            device,
            loader,
            optimizer,
        )

    save_model(save_best=False)
