import ase
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, radius_graph, GPSConv
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

activation_dict = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU, "Swish": nn.SiLU}


class InteractionMLP(nn.Module):
    def __init__(
        self,
        emb_dim,
        dropout=0.0,
        num_layers=2,
        interaction_agg="cat",
        normalizer=nn.Identity,
        initializer="glorot",
        activation="GELU",
    ):
        super(InteractionMLP, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.interaction_agg = interaction_agg

        act = activation_dict[activation]

        layers = []
        emb_dim = emb_dim * 2 if interaction_agg == "cat" else emb_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(normalizer(emb_dim))
            layers.append(act())

        self.layers = nn.Sequential(*layers)
        self.initializer = initializer

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                apply_init(self.initializer)(layer.weight)
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
        self.interactor_residual = args.interactor_residual
        self.interaction_agg = interaction_agg
        self.interaction_rep_2d = interaction_rep_2d
        self.interaction_rep_3d = interaction_rep_3d
        self.JK = args.JK

        self.model_2d = args.model_2d
        self.model_3d = args.model_3d

        self.dropout = dropout
        self.emb_dim = emb_dim

        self.atom_encoder_2d = AtomEncoder(emb_dim=emb_dim)
        self.atom_encoder_3d = None

        self.atom_encoder_3d = nn.Embedding(num_node_class, emb_dim)
        self.atomic_mass = torch.from_numpy(ase.data.atomic_masses).to(device)

        self.mlp_3d = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), ShiftedSoftplus(), nn.Linear(emb_dim, emb_dim)
        )
        for layer in self.mlp_3d:
            if isinstance(layer, nn.Linear):
                apply_init(args.initialization)(layer.weight)
                nn.init.zeros_(layer.bias)

        block_3d = SchNetBlock

        self.blocks_3d = nn.ModuleList(
            [block_3d(hidden_channels=emb_dim) for _ in range(num_interaction_blocks)]
        )

        self.blocks_2d = nn.ModuleList(
            [
                Block2D(args, model_2d, initializer=args.initialization)
                for _ in range(num_interaction_blocks)
            ]
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
                        activation=args.interactor_activation,
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
                activation=args.interactor_activation,
            )

            self.interactor = interactor

        self.reset_parameters()

    def reset_parameters(self):
        for block in self.blocks_3d:
            block.reset_parameters()
        for block in self.blocks_2d:
            block.reset_parameters()
        if self.diff_interactor_per_block:
            for interactor in self.interactors:
                interactor.reset_parameters()
        else:
            self.interactor.reset_parameters()
        for layer in self.mlp_3d:
            if isinstance(layer, nn.Linear):
                apply_init(self.args.initialization)(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_3d(
        self,
        x,
        positions,
        batch,
    ):
        orig_x = x
        if self.model_3d == "SchNet":
            x_3d = x
            if x.dim() != 1:
                x_3d = x[:, 0]

            assert x_3d.dim() == 1 and x_3d.dtype == torch.long
            batch = torch.zeros_like(x) if batch is None else batch

        if self.args.transfer:
            with torch.no_grad():
                x_2d = self.atom_encoder_2d(orig_x)

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

            if self.args.transfer:
                combined_x = self.interactors[i](x, x_2d)
                x_2d, x = combined_x.split(self.emb_dim, dim=-1)

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
        orig_x = x
        x = self.atom_encoder_2d(x)
        if self.args.transfer:
            with torch.no_grad():
                x_3d = self.atom_encoder_3d(orig_x[:, 0])

        h_list_2d = [x]
        prev = x
        for i in range(self.num_interaction_blocks):
            x = self.blocks_2d[i](x, edge_index, edge_attr)
            x = self.dropouts[i](x)
            # x = self.norm_2d[i](x)
            # x = F.relu(x)

            if self.residual:
                x = x + prev

            if self.args.transfer:
                combined_x = self.interactors[i](x, x_3d)
                x, x_3d = combined_x.split(self.emb_dim, dim=-1)

            h_list_2d.append(x)
            prev = x

        if self.interaction_rep_2d == "vnode":
            pass
        elif self.interaction_rep_2d == "mean":
            x = global_mean_pool(x, batch)
        elif self.interaction_rep_2d == "sum":
            x = global_add_pool(x, batch)

        if self.JK == "last":
            return x
        # elif self.JK in ("sum", "mean"):
        elif self.JK == "sum":
            pooler = global_add_pool
        elif self.JK == "mean":
            pooler = global_mean_pool

        pooled = [pooler(h, batch) for h in h_list_2d]
        outs = torch.stack(pooled)
        final = torch.cat([outs, x.unsqueeze(0)], dim=0)

        if self.JK == "sum":
            return final.sum(dim=0)
        elif self.JK == "mean":
            return final.mean(dim=0)

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

        if self.model_3d == "SchNet":
            x_3d = x
            if x.dim() != 1:
                x_3d = x[:, 0]

            assert x_3d.dim() == 1 and x_3d.dtype == torch.long
            batch = torch.zeros_like(x) if batch is None else batch

            x_3d = self.atom_encoder_3d(x_3d)

        prev_3d = x_3d

        interactor_residual_3d, interactor_residual_2d = None, None

        for i in range(self.num_interaction_blocks):
            x_2d = self.blocks_2d[i](x_2d, edge_index, edge_attr)
            x_2d = self.dropouts[i](x_2d)
            x_2d = self.norm_2d[i](x_2d)

            if self.residual:
                x_2d = x_2d + prev_2d

            if self.model_3d == "SchNet":
                x_3d = self.blocks_3d[i](x_3d, positions, batch)
            x_3d = self.dropouts[i](x_3d)
            x_3d = self.norm_3d[i](x_3d)

            if self.residual:
                x_3d = x_3d + prev_3d

            prev_2d = x_2d
            prev_3d = x_3d

            if (
                self.interactor_residual
                and interactor_residual_2d is not None
                and interactor_residual_3d is not None
            ):
                x_2d_int = x_2d + interactor_residual_2d
                x_3d_int = x_3d + interactor_residual_3d
            else:
                x_2d_int = x_2d
                x_3d_int = x_3d

            if self.diff_interactor_per_block:
                interaction = self.interactors[i](x_2d_int, x_3d_int)
            else:
                interaction = self.interactor(x_2d_int, x_3d_int)

            x_2d, x_3d = torch.split(interaction, self.emb_dim, dim=-1)

            if self.interactor_residual:
                interactor_residual_2d = x_2d
                interactor_residual_3d = x_3d

            if i == self.num_interaction_blocks - 1:
                x_3d = self.mlp_3d(x_3d)

            midstream_outs_2d.append(x_2d)
            midstream_outs_3d.append(x_3d)

        if self.interaction_rep_2d == "mean":
            rep_2d = global_mean_pool(x_2d, batch)
        elif self.interaction_rep_2d == "sum":
            rep_2d = global_add_pool(x_2d, batch)

        if self.interaction_rep_3d == "mean":
            rep_3d = global_mean_pool(x_3d, batch)
        elif self.interaction_rep_3d == "sum":
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

        if self.JK == "last":
            return x
        elif self.JK in ("sum", "mean"):
            means_2d = [
                global_mean_pool(midstream_2d, batch)
                for midstream_2d in midstream_outs_2d
            ]
            means_3d = [
                global_mean_pool(midstream_3d, batch)
                for midstream_3d in midstream_outs_3d
            ]

            outs_2d = torch.stack(means_2d)
            outs_3d = torch.stack(means_3d)

            outs = torch.cat([outs_2d, outs_3d], dim=-1)
            final = torch.cat([outs, x.unsqueeze(0)], dim=0)

            if self.JK == "sum":
                return final.sum(dim=0)
            elif self.JK == "mean":
                return final.mean(dim=0)
        else:
            return x


class Block2D(nn.Module):
    def __init__(self, args, gnn_type="GIN", initializer="glorot_uniform"):
        super(Block2D, self).__init__()
        self.emb_dim = args.emb_dim
        self.gnn_type = gnn_type

        layer = None
        if gnn_type == "GIN":
            layer = GINBlock(self.emb_dim, initializer=initializer)
        elif gnn_type == "GAT":
            layer = GATConv(self.emb_dim, heads=args.gat_heads)
        elif gnn_type == "GCN":
            layer = GCNConv(self.emb_dim)
        elif gnn_type == "GraphSAGE":
            layer = GraphSAGEConv(self.emb_dim, self.emb_dim)
        elif gnn_type == "Transformer":
            layer = TransformerConv(self.emb_dim, heads=args.transformer_heads)
        elif gnn_type == "GPS":
            layer = GPSConv(
                self.emb_dim, GINBlock(self.emb_dim, initializer=initializer), heads=2
            )
        else:
            raise ValueError("Invalid GNN type")

        self.layer = layer

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr=edge_attr)
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

    def reset_parameters(self):
        self.interaction.reset_parameters()

    def forward(self, h, pos, batch=None):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        h = self.interaction(h, edge_index, edge_weight, edge_attr)

        return h
