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

from Geom3D.datasets import Molecule3DDataset
from Geom3D.models import GNN, SchNet, EGNN
from config import args
from ogb.utils.features import get_atom_feature_dims


CE_criterion = nn.CrossEntropyLoss()

ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0


def compute_accuracy(pred, target):
    return float(
        torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()
    ) / len(pred)


def do_2D_masking(batch, node_repr, molecule_atom_masking_model, masked_atom_indices):
    target = batch.x[masked_atom_indices][:, 0].detach()
    node_pred = molecule_atom_masking_model(node_repr[masked_atom_indices])
    loss = CE_criterion(node_pred, target)
    acc = compute_accuracy(node_pred, target)
    return loss, acc


def perturb(x, positions, mu, sigma):
    x_perturb = x

    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(
        device
    )

    return x_perturb, positions_perturb


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            output_model_path = os.path.join(
                args.output_model_dir, "model_complete.pth"
            )
            saver_dict = {
                "model_2D": molecule_model_2D.state_dict(),
                "model_2D_pos": molecule_model_2D.state_dict(),
                "model_3D": molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, output_model_path)

        else:
            output_model_path = os.path.join(
                args.output_model_dir, "model_complete_final.pth"
            )
            saver_dict = {
                "model_2D": molecule_model_2D.state_dict(),
                "model_2D_pos": molecule_model_2D.state_dict(),
                "model_3D": molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, output_model_path)
    return


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
    ],  # hyperparams corresponding to invariant loss, equivariant loss, hybrid loss
):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_2D_pos.train()
    molecule_model_3D.eval()

    CL_loss_accum, CL_acc_accum = 0, 0

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

        ### invariant loss
        mol_2d_repr = pool(node_2D_repr, batch.batch, pool=args.pooling)
        mol_2d_repr = graph_pred_linear(mol_2d_repr)

        mol_3d_repr = pool(node_3D_repr, batch.batch, pool=args.pooling)

        invariant_loss = balances[0] * NTXentLoss(mol_2d_repr, mol_3d_repr)

        ### equivariant loss

        ### combined loss

        loss = invariant_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)

    temp_loss = 0  # todo

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print("CL Loss: {:.5f}\tCL Acc: {:.5f}".format(CL_loss_accum, CL_acc_accum))
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
            in_node_nf=num_node_classes,
            in_edge_nf=num_edge_classes,
            hidden_nf=args.emb_dim_egnn,
            n_layers=args.n_layers_egnn,
            positions_weight=args.positions_weight_egnn,
            attention=args.attention_egnn,
            node_attr=True,
        )

    else:
        raise NotImplementedError("Model {} not included.".format(args.model_3d))

    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    graph_pred_linear = nn.Sequential(
        nn.Linear(intermediate_dim, intermediate_dim // 2),
        nn.BatchNorm1d(intermediate_dim // 2),
        nn.ReLU(),
        nn.Linear(intermediate_dim // 2, intermediate_dim // 2),
    ).to(device)

    model_weight = torch.load(args.input_model_file_3d)
    if "model_3D" in model_weight:
        molecule_model_3D.load_state_dict(model_weight["model_3D"])

    model_param_group = []
    model_param_group.append({"params": graph_pred_linear.parameters(), "lr": args.lr})
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
