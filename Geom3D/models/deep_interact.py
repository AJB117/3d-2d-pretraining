import ase
import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    radius_graph,
)
from .encoders import AtomEncoder
from .molecule_gnn_model import (
    GINBlock,
    GATConv,
    GCNConv,
    GraphSAGEConv,
    TransformerConv,
)
from .schnet import InteractionBlock, GaussianSmearing, ShiftedSoftplus
from runners.util import apply_init


class InteractionMLP(nn.Module):
    def __init__(
        self,
        emb_dim,
        dropout=0.0,
        num_layers=2,
        interaction_agg="cat",
        normalizer=nn.Identity,
        initializer="glorot",
    ):
        super(InteractionMLP, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.interaction_agg = interaction_agg

        layers = []
        emb_dim = emb_dim * 2 if interaction_agg == "cat" else emb_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(normalizer(emb_dim))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # nn.init.xavier_uniform_(layer.weight)
                apply_init(initializer)(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, rep_2d, rep_3d):
        if self.interaction_agg == "cat":
            x = torch.cat([rep_2d, rep_3d], dim=-1)
        elif self.interaction_agg == "sum":
            x = rep_2d + rep_3d
        elif self.interaction_agg == "mean":
            x = (rep_2d + rep_3d) / 2
        else:
            raise ValueError("Invalid interaction aggregation method")

        return self.layers(x)


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
        device="cuda",
        batch_norm=True,
        layer_norm=False,
        dropout=0.0,
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

            self.lin1 = nn.Linear(emb_dim, emb_dim)
            self.schnet_act = ShiftedSoftplus()
            self.lin2 = nn.Linear(emb_dim, emb_dim)

            block_3d = SchNetBlock

        self.blocks_3d = nn.ModuleList(
            [block_3d(hidden_channels=emb_dim) for _ in range(num_interaction_blocks)]
        )

        self.blocks_2d = nn.ModuleList(
            [Block2D(args, model_2d) for _ in range(num_interaction_blocks)]
        )

        if layer_norm:
            normalizer = nn.LayerNorm
        elif batch_norm:
            normalizer = nn.BatchNorm1d
        else:
            normalizer = nn.Identity

        self.norm_2d = nn.ModuleList(
            [normalizer(emb_dim) for _ in range(num_interaction_blocks)]
        )
        self.norm_3d = nn.ModuleList(
            [normalizer(emb_dim) for _ in range(num_interaction_blocks)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_interaction_blocks)]
        )

        self.final_pool = final_pool

        self.diff_interactor_per_block = args.diff_interactor_per_block

        if self.diff_interactor_per_block:
            self.interactors = nn.ModuleList(
                [
                    InteractionMLP(
                        emb_dim,
                        dropout=dropout,
                        num_layers=2,
                        interaction_agg=interaction_agg,
                        normalizer=normalizer,
                        initializer=args.initialization,
                    ).to(device)
                    for _ in range(num_interaction_blocks)
                ]
            )
        else:
            interactor = InteractionMLP(
                emb_dim,
                dropout=dropout,
                num_layers=2,
                interaction_agg=interaction_agg,
                normalizer=normalizer,
                initializer=args.initialization,
            )

            self.interactor = interactor

    def forward_3d(
        self,
        x,
        positions,
        batch,
    ):
        if self.model_3d == "SchNet":
            x_3d = x
            if x.dim() != 1:
                x_3d = x[:, 0]

            assert x_3d.dim() == 1 and x_3d.dtype == torch.long
            batch = torch.zeros_like(x) if batch is None else batch

        x = self.atom_encoder_3d(x_3d)
        prev = x
        for i in range(self.num_interaction_blocks):
            if self.model_3d == "SchNet":
                x = self.blocks_3d[i](x, positions, batch)

            x = self.dropouts[i](x)
            x = self.norm_3d[i](x)
            # x = F.relu(x)

            if self.residual:
                x = x + prev

            prev = x

        if self.interaction_rep_3d == "com":
            pass
        elif self.interaction_rep_3d == "mean":
            x = global_mean_pool(x, batch)
        elif self.interaction_rep_3d == "sum":
            x = global_add_pool(x, batch)

        return x

    def forward_2d(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):
        x = self.atom_encoder_2d(x)
        prev = x
        for i in range(self.num_interaction_blocks):
            x = self.blocks_2d[i](x, edge_index, edge_attr)
            x = self.dropouts[i](x)
            # x = self.norm_2d[i](x)
            # x = F.relu(x)

            if self.residual:
                x = x + prev

            prev = x

        if self.interaction_rep_2d == "vnode":
            pass
        elif self.interaction_rep_2d == "mean":
            x = global_mean_pool(x, batch)
        elif self.interaction_rep_2d == "sum":
            x = global_add_pool(x, batch)

        return x

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        positions,
        batch,
        require_midstream=False,
    ):
        midstream_outs_2d, midstream_outs_3d = [], []

        x_2d = self.atom_encoder_2d(x)
        prev_2d = x_2d

        # find the virtual nodes out of the batch
        # num_nodes_per_graph = scatter_add(
        #     torch.ones_like(positions[:, 0]), batch, dim=0
        # ).type(torch.long)
        # cum_indices_per_graph = torch.cumsum(num_nodes_per_graph, dim=0).type(
        #     torch.long
        # )

        # virt_mask = torch.zeros_like(positions[:, 0], dtype=torch.bool)
        # virt_mask[cum_indices_per_graph - 1] = True

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
            # x_2d = F.relu(x_2d)

            if self.residual:
                x_2d = x_2d + prev_2d

            if self.model_3d == "SchNet":
                x_3d = self.blocks_3d[i](x_3d, positions, batch)
            x_3d = self.dropouts[i](x_3d)
            x_3d = self.norm_3d[i](x_3d)

            # x_3d = F.relu(x_3d)

            if self.residual:
                x_3d = x_3d + prev_3d

            # virt_emb_2d = x_2d[virt_mask]
            # virt_emb_3d = x_3d[virt_mask]

            # if self.interaction_agg == "cat":
            #     interaction = torch.cat([virt_emb_2d, virt_emb_3d], dim=-1)
            # elif self.interaction_agg == "sum":
            #     interaction = virt_emb_2d + virt_emb_3d
            # elif self.interaction_agg == "mean":
            #     interaction = (virt_emb_2d + virt_emb_3d) / 2

            # mean_2d = global_mean_pool(x_2d, batch)
            # mean_3d = global_mean_pool(x_3d, batch)

            prev_2d = x_2d
            prev_3d = x_3d

            # interaction = torch.cat([x_2d, x_3d], dim=-1)
            if self.diff_interactor_per_block:
                interaction = self.interactors[i](x_2d, x_3d)
            else:
                interaction = self.interactor(x_2d, x_3d)

            # virt_emb_2d, virt_emb_3d = torch.split(interaction, self.emb_dim, dim=-1)
            x_2d, x_3d = torch.split(interaction, self.emb_dim, dim=-1)

            midstream_outs_2d.append(x_2d)
            midstream_outs_3d.append(x_3d)

            # x_2d[virt_mask] = virt_emb_2d
            # x_3d[virt_mask] = virt_emb_3d

        if self.model_3d == "SchNet":
            x_3d = self.lin1(x_3d)
            x_3d = self.schnet_act(x_3d)
            x_3d = self.lin2(x_3d)

        if self.interaction_rep_2d == "vnode":
            rep_2d = x_2d[virt_mask]
        elif self.interaction_rep_2d == "mean":
            # rep_2d = x_2d.mean(dim=0)
            rep_2d = global_mean_pool(x_2d, batch)
        elif self.interaction_rep_2d == "sum":
            # rep_2d = x_2d.sum(dim=0)
            rep_2d = global_add_pool(x_2d, batch)

        if self.interaction_rep_3d in ("com", "const_radius"):
            rep_3d = x_3d[virt_mask]
        elif self.interaction_rep_3d == "mean":
            # rep_3d = x_3d[~virt_mask].mean(dim=0, keepdim=True)
            # rep_3d = x_3d.mean(dim=0)
            rep_3d = global_mean_pool(x_3d, batch)
        elif self.interaction_rep_3d == "sum":
            # rep_3d = x_3d.sum(dim=0)
            rep_3d = global_add_pool(x_3d, batch)

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

        if require_midstream:
            return x, midstream_outs_2d, midstream_outs_3d

        return x


class Block2D(nn.Module):
    def __init__(self, args, gnn_type="GIN"):
        super(Block2D, self).__init__()
        self.emb_dim = args.emb_dim
        self.gnn_type = gnn_type

        layer = None
        if gnn_type == "GIN":
            layer = GINBlock(self.emb_dim)
        elif gnn_type == "GAT":
            layer = GATConv(self.emb_dim, heads=args.gat_heads)
        elif gnn_type == "GCN":
            layer = GCNConv(self.emb_dim)
        elif gnn_type == "GraphSAGE":
            layer = GraphSAGEConv(self.emb_dim, self.emb_dim)
        elif gnn_type == "Transformer":
            layer = TransformerConv(self.emb_dim, heads=args.transformer_heads)
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
        num_gaussians=50,
        cutoff=10.0,
    ):
        super(SchNetBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
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
