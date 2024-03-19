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
from .schnet import InteractionBlock, GaussianSmearing
from torch_scatter import scatter
import ase


class Interactor(nn.Module):
    def __init__(
        self,
        args,
        num_interaction_blocks,
        final_pool,
        emb_dim,
        model_2d="GIN",
        model_3d="SchNet",
        residual=True,
        interaction_agg="cat",
        interaction_rep_2d="vnode",
        interaction_rep_3d="com",
        num_node_class=119,
        device="cpu",
        bnorm=True,
        lnorm=False,
        dropout=0.0
    ):
        super(Interactor, self).__init__()
        self.args = args
        self.num_interaction_blocks = num_interaction_blocks
        self.residual = residual
        self.interaction_agg = interaction_agg
        self.interaction_rep_2d = interaction_rep_2d
        self.interaction_rep_3d = interaction_rep_3d

        self.model_2d = args.model_2d
        self.model_3d = args.model_3d

        self.dropout = dropout
        self.emb_dim = emb_dim

        self.atom_encoder_2d = AtomEncoder(emb_dim=emb_dim)
        self.atom_encoder_3d = None

        if model_3d == "SchNet":
            self.atom_encoder_3d = nn.Embedding(num_node_class, emb_dim)
            self.atomic_mass = torch.from_numpy(ase.data.atomic_masses).to(device)

        self.blocks_2d = nn.ModuleList(
            [Block2D(args, model_2d) for _ in range(num_interaction_blocks)]
        )

        if model_3d == "SchNet":
            block_3d = SchNetBlock(hidden_channels=emb_dim)

        self.blocks_3d = nn.ModuleList(
            [block_3d for _ in range(num_interaction_blocks)]
        )

        if lnorm:
            normalizer = nn.LayerNorm(emb_dim)
        elif bnorm:
            normalizer = nn.BatchNorm1d(emb_dim)
        else:
            normalizer = nn.Identity()

        self.norm_2d = nn.ModuleList(
            [normalizer for _ in range(num_interaction_blocks)]
        )
        self.norm_3d = nn.ModuleList(
            [normalizer for _ in range(num_interaction_blocks)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_interaction_blocks)]
        )

        self.final_pool = final_pool

        twice_dim = args.emb_dim * 2
        if interaction_agg == "cat":
            interactor = nn.Sequential(
                nn.Linear(twice_dim, twice_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(twice_dim),
                nn.ReLU(),
                nn.Linear(twice_dim, twice_dim),
            )
        elif interaction_agg in ("sum", "mean"):
            interactor = nn.Sequential(
                nn.Linear(args.emb_dim, twice_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(twice_dim),
                nn.ReLU(),
                nn.Linear(twice_dim, twice_dim),
            )
        
        # use xavier initialization for the interactor
        for layer in interactor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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

        # find the virtual nodes out of the batch
        num_nodes_per_graph = scatter_add(
            torch.ones_like(positions[:, 0]), batch, dim=0
        ).type(torch.long)
        cum_indices_per_graph = torch.cumsum(num_nodes_per_graph, dim=0).type(
            torch.long
        )

        virt_mask = torch.zeros_like(positions[:, 0], dtype=torch.bool)
        virt_mask[cum_indices_per_graph - 1] = True

        if self.model_3d == "SchNet":
            x_3d = x
            if x.dim() != 1:
                x_3d = x[:, 0]

            assert x_3d.dim() == 1 and x_3d.dtype == torch.long
            batch = torch.zeros_like(x) if batch is None else batch

            x_3d = self.atom_encoder_3d(x_3d)

        prev_3d = x_3d

        for i in range(self.num_interaction_blocks):
            x_2d = self.blocks_2d[i](x_2d, edge_index, edge_attr)
            x_2d = self.dropouts[i](x_2d)
            x_2d = self.norm_2d[i](x_2d)
            x_2d = F.relu(x_2d)

            if self.residual:
                x_2d = x_2d + prev_2d

            if self.model_3d == "SchNet":
                x_3d = self.blocks_3d[i](x_3d, positions, batch)
            x_3d = self.dropouts[i](x_3d)
            x_3d = self.norm_3d[i](x_3d)
            x_3d = F.relu(x_3d)

            if self.residual:
                x_3d = x_3d + prev_3d

            virt_emb_2d = x_2d[virt_mask]
            virt_emb_3d = x_3d[virt_mask]

            if self.interaction_agg == "cat":
                interaction = torch.cat([virt_emb_2d, virt_emb_3d], dim=-1)
            elif self.interaction_agg == "sum":
                interaction = virt_emb_2d + virt_emb_3d
            elif self.interaction_agg == "mean":
                interaction = (virt_emb_2d + virt_emb_3d) / 2

            interaction = self.interactor(interaction)
            virt_emb_2d, virt_emb_3d = torch.split(interaction, self.emb_dim, dim=-1)

            x_2d[virt_mask] = virt_emb_2d
            x_3d[virt_mask] = virt_emb_3d

            prev_2d = x_2d
            prev_3d = x_3d

        if self.interaction_rep_2d == "vnode":
            rep_2d = x_2d[virt_mask]
        elif self.interaction_rep_2d == "mean":
            rep_2d = x_2d.mean(dim=0, keepdim=True)
        elif self.interaction_rep_2d == "sum":
            rep_2d = x_2d.sum(dim=0, keepdim=True)
        
        if self.interaction_rep_3d in ("com", "const_radius"):
            rep_3d = x_3d[virt_mask]
        elif self.interaction_rep_3d == "mean":
            rep_3d = x_3d[~virt_mask].mean(dim=0, keepdim=True)
        elif self.interaction_rep_3d == "sum":
            rep_3d = x_3d.sum(dim=0, keepdim=True)

        if self.final_pool == "attention":
            pass  #! TODO
        elif self.final_pool == "cat":
            x = torch.cat([rep_2d, rep_3d], dim=-1)
        elif self.final_pool == "mean":
            x = (rep_2d + rep_3d) / 2
        elif self.final_pool == "sum":
            x = rep_2d + rep_3d
        else:
            raise ValueError("Invalid final pooling method")

        # x = self.interactor(x)

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
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interaction = InteractionBlock(
            hidden_channels, num_gaussians, num_filters, cutoff
        )

    def forward(self, h, pos, batch=None):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        h = self.interaction(h, edge_index, edge_weight, edge_attr)

        return h
