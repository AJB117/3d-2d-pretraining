import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    radius_graph,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
from .encoders import AtomEncoder
from typing import List
from .molecule_gnn_model import GINConv, GATConv, GCNConv, GraphSAGEConv
from .schnet import InteractionBlock
from torch_scatter import scatter
import ase


class Interactor(nn.Module):
    def __init__(
        self,
        args,
        num_interaction_blocks,
        final_pool,
        emb_dim,
        gnn_2d_type="GIN",
        model_3d="SchNet",
        residual=True,
        interaction_agg="cat",
        num_node_class=119,
        device="cpu",
    ):
        super(Interactor, self).__init__()
        self.args = args
        self.atom_encoder = AtomEncoder(emb_dim=emb_dim)
        self.num_interaction_blocks = num_interaction_blocks
        self.residual = residual
        self.interaction_agg = interaction_agg
        self.model_2d = args.model_2d
        self.model_3d = args.model_3d

        self.atom_encoder_2d = AtomEncoder(emb_dim=emb_dim)
        self.atom_encoder_3d = None

        if model_3d == "SchNet":
            self.atom_encoder_3d = nn.Embedding(num_node_class, emb_dim)
            self.atomic_mass = torch.from_numpy(ase.data.atomic_masses).to(device)

        self.blocks_2d = nn.ModuleList(
            [Block2D(args, gnn_2d_type) for _ in range(num_interaction_blocks)]
        )

        if model_3d == "SchNet":
            block_3d = SchNetBlock

        self.blocks_3d = nn.ModuleList(
            [block_3d() for _ in range(num_interaction_blocks)]
        )

        self.bn_2d = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_interaction_blocks)]
        )
        self.bn_3d = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_interaction_blocks)]
        )

        self.final_pool = final_pool

        twice_dim = args.emb_dim * 2
        if interaction_agg == "cat":
            interactor = nn.Sequential(
                nn.Linear(twice_dim, twice_dim),
                nn.BatchNorm1d(twice_dim),
                nn.ReLU(),
                nn.Linear(twice_dim, twice_dim),
            )
        elif interaction_agg in ("sum", "add"):
            interactor = nn.Sequential(
                nn.Linear(args.emb_dim, twice_dim),
                nn.BatchNorm1d(twice_dim),
                nn.ReLU(),
                nn.Linear(twice_dim, twice_dim),
            )

        self.interactor = interactor

    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, positions, batch = (
                argv[0],
                argv[1],
                argv[2],
                argv[3],
                argv[4],
            )
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, positions, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.positions,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        x_2d = self.atom_encoder_2d(x)
        prev_2d = x_2d

        if self.model_3d == "SchNet":
            if x.dim() != 1:
                x = x[
                    :, 0
                ]  # avoid rebuilding the dataset when use_pure_atomic_num is False
            assert x.dim() == 1 and x.dtype == torch.long
            batch = torch.zeros_like(x) if batch is None else batch

            if self.args.interaction_rep_3d == "com":
                mass = self.atomic_mass[x[:-1]].view(-1, 1)
                c = scatter(mass * positions[:-1], batch[:-1], dim=0) / scatter(
                    mass, batch[:-1], dim=0
                )
                pdb.set_trace()
                positions[-1] = c

            x_3d = self.atom_encoder_3d(x)

        prev_3d = x_3d

        for i in range(self.num_interaction_blocks):
            x_2d = self.blocks_2d[i](x_2d, edge_index, edge_attr)
            x_2d = F.relu(x_2d)
            x_2d = self.bn_2d[i](x_2d)

            if self.residual:
                x_2d = x_2d + prev_2d

            if self.model_3d == "SchNet":
                x_3d = self.blocks_3d[i](x_3d, positions, batch)
            x_3d = F.relu(x_3d)
            x_3d = self.bn_3d[i](x_3d)

            if self.residual:
                x_3d = x_3d + prev_3d

            num_nodes_per_batch = scatter_add(torch.ones_like(x_2d[:, 0]), batch, dim=0)
            virt_emb_2d = x_2d[num_nodes_per_batch - 1]
            virt_emb_3d = x_3d[num_nodes_per_batch - 1]

            if self.interaction_agg == "cat":
                interaction = torch.cat([virt_emb_2d, virt_emb_3d], dim=-1)
            elif self.interaction_agg == "sum":
                interaction = virt_emb_2d + virt_emb_3d
            elif self.interaction_agg == "mean":
                interaction = (virt_emb_2d + virt_emb_3d) / 2

            interaction = self.interactor(interaction)
            virt_emb_2d, virt_emb_3d = torch.split(interaction, self.emb_dim, dim=-1)

            x_2d[num_nodes_per_batch - 1] = virt_emb_2d
            x_3d[num_nodes_per_batch - 1] = virt_emb_3d

        if self.final_pool == "attention":
            pass  #! TODO
        elif self.final_pool == "cat":
            x = torch.cat([virt_emb_2d, virt_emb_3d], dim=-1)
        elif self.final_pool == "mean":
            x = (virt_emb_2d + virt_emb_3d) / 2
        else:
            raise ValueError("Invalid final pooling method")

        x = self.interactor(x)

        return x


class Block2D(nn.Module):
    def __init__(self, args, gnn_type="GIN"):
        super(Block2D, self).__init__()
        self.emb_dim = args.emb_dim
        self.gnn_type = gnn_type

        layer = None
        if gnn_type == "GIN":
            layer = GINConv(self.emb_dim)
        elif gnn_type == "GAT":
            layer = GATConv(self.emb_dim, self.emb_dim // 2, heads=4)
        elif gnn_type == "GCN":
            layer = GCNConv(self.emb_dim, self.emb_dim)
        elif gnn_type == "GraphSAGE":
            layer = GraphSAGEConv(self.emb_dim, self.emb_dim)
        else:
            raise ValueError("Invalid GNN type")

        self.layer = layer

    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr)
        return x


class SchNetBlock(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        node_class=120,
    ):
        super(SchNetBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        self.interaction = InteractionBlock(
            hidden_channels, num_gaussians, num_filters, cutoff
        )

    def forward(self, z, pos, batch=None, interaction_rep="com"):
        if z.dim() != 1:
            z = z[
                :, 0
            ]  # avoid rebuilding the dataset when use_pure_atomic_num is False
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if interaction_rep == "com":
            mass = self.atomic_mass[z[:-1]].view(-1, 1)
            c = scatter(mass * pos[:-1], batch[:-1], dim=0) / scatter(
                mass, batch[:-1], dim=0
            )
            pos[-1] = c

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        h = self.interaction(h, edge_index, edge_weight, edge_attr)

        return h
