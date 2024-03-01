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
from torch_geometric.nn import global_mean_pool

from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius
from Geom3D.models import GNN, SchNet, PaiNN, PNA
from Geom3D.models.MoleculeSDE import (
    SDEModel2Dto3D_01,
    SDEModel2Dto3D_02,
    SDEModel3Dto2D_node_adj_dense,
    SDEModel3Dto2D_node_adj_dense_02,
    SDEModel3Dto2D_node_adj_dense_03,
)
from util import dual_CL, get_random_indices
from config import args


CE_criterion = nn.CrossEntropyLoss()
from ogb.utils.features import get_atom_feature_dims

ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0


def ntxent_loss(z1, z2, temperature=0.1):
    sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

    z1_abs = z1.norm(dim=1)
    z2_abs = z2.norm(dim=1)
    sim_matrix = sim_matrix / (torch.einsum("i,j->ij", z1_abs, z2_abs) + 1e-8)

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = torch.diagonal(sim_matrix)
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()

    return loss


def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_2D_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        mol_2d_repr = molecule_readout_func(node_2D_repr, batch.batch)

        if args.model_3d == "SchNet":
            # batch.x = batch.x.unsqueeze(1)
            _, node_3D_repr = molecule_model_3D(
                batch.x[:, 0], batch.positions, batch.batch, return_latent=True
            )

        mol_3d_repr = molecule_readout_func(node_3D_repr, batch.batch)
        mol_3d_repr = up_project_3d_network(mol_3d_repr)

        loss = ntxent_loss(mol_2d_repr, mol_3d_repr, temperature=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
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
    node_class = 119

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

    if args.model_3d == "PaiNN":
        data_root = "{}_{}".format(data_root, args.PaiNN_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.PaiNN_radius_cutoff,
            mask_ratio=args.SSL_masking_ratio,
            remove_center=True,
            use_extend_graph=args.use_extend_graph,
        )

    if args.dataset == "QM9":
        indices = get_random_indices(len(dataset))
        pretrain_indices = indices[:50000]
        dataset = dataset[pretrain_indices]

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # set up model
    if args.gnn_type == "PNA":
        molecule_model_2D = PNA(
            hidden_dim=args.emb_dim,
            target_dim=args.emb_dim,
            aggregators=["sum", "mean", "min", "max"],
            scalers=["identity", "amplification", "attenuation"],
            propagation_depth=args.num_layer,
            posttrans_layers=2,
            pretrans_layers=1,
            readout_layers=2,
            readout_aggregators=["min", "max", "mean"],
            avg_d=PNA.get_degree_histogram(loader),
        ).to(device)
    else:
        molecule_model_2D = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)
    molecule_readout_func = global_mean_pool

    print("Using 3d model\t", args.model_3d)
    if args.model_3d == "SchNet":
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=node_class,
        ).to(device)
    elif args.model_3d == "PaiNN":
        molecule_model_3D = PaiNN(
            n_atom_basis=args.emb_dim,
            n_interactions=args.PaiNN_n_interactions,
            n_rbf=args.PaiNN_n_rbf,
            cutoff=args.PaiNN_radius_cutoff,
            max_z=node_class,
            n_out=1,
            readout=args.PaiNN_readout,
        ).to(device)
    else:
        raise NotImplementedError("Model {} not included.".format(args.model_3d))

    up_project_3d_network = nn.Sequential(
        nn.Linear(args.emb_dim, args.emb_dim),
        nn.BatchNorm1d(args.emb_dim),
        nn.ReLU(),
        nn.Linear(args.emb_dim, args.emb_dim),
    ).to(device)

    model_param_group = []
    model_param_group.append(
        {"params": molecule_model_2D.parameters(), "lr": args.lr * args.gnn_2d_lr_scale}
    )
    model_param_group.append(
        {"params": molecule_model_3D.parameters(), "lr": args.lr * args.gnn_3d_lr_scale}
    )
    model_param_group.append(
        {"params": up_project_3d_network.parameters(), "lr": args.lr}
    )

    print(
        "number of parameters for 2d model: {}".format(
            sum(p.numel() for p in molecule_model_2D.parameters())
        )
    )
    print(
        "number of parameters for 3d model: {}".format(
            sum(p.numel() for p in molecule_model_3D.parameters())
        )
    )

    print("pretraining on {} samples".format(len(dataset)))

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10
    SDE_coeff_contrastive_oriGINal = args.SDE_coeff_contrastive
    args.SDE_coeff_contrastive = 0

    for epoch in range(1, args.epochs + 1):
        if epoch > args.SDE_coeff_contrastive_skip_epochs:
            args.SDE_coeff_contrastive = SDE_coeff_contrastive_oriGINal
        print("epoch: {}".format(epoch))
        train(args, molecule_model_2D, device, loader, optimizer)
