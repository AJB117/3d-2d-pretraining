import pdb
import torch
import torch.nn as nn
from .self_attention import SelfAttentionBlock
from torchdrug.layers.functional.functional import (
    padded_to_variadic,
    variadic_to_padded,
)

from runners.util import apply_init, activation_dict


class InteractionMHA(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads=4,
        dropout=0.0,
        interaction_agg="cat",
        normalizer=nn.Identity,
        initializer="glorot",
        activation="GELU",
    ):
        super(InteractionMHA, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.interaction_agg = interaction_agg

        act = activation_dict[activation]

        self.initializer = initializer

        layers_2d_3d = []
        layers_3d_2d = []

        self.mha_2d_3d = SelfAttentionBlock(emb_dim, 1, dropout=0.2)
        self.mha_3d_2d = SelfAttentionBlock(emb_dim, 1, dropout=0.2)

        for _ in range(1):
            layers_2d_3d.append(nn.Linear(emb_dim, emb_dim))
            layers_3d_2d.append(nn.Linear(emb_dim, emb_dim))

            layers_2d_3d.append(normalizer(emb_dim))
            layers_3d_2d.append(normalizer(emb_dim))

            layers_2d_3d.append(act())
            layers_3d_2d.append(act())

        self.layers_2d_3d = nn.Sequential(*layers_2d_3d)
        self.layers_3d_2d = nn.Sequential(*layers_3d_2d)

    def reset_parameters(self):
        for layer in self.layers_2d_3d:
            if isinstance(layer, nn.Linear):
                apply_init(self.initializer)(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in [self.mha_2d_3d, self.mha_3d_2d]:
            apply_init(self.initializer)(layer.query.weight)
            nn.init.zeros_(layer.query.bias)
            apply_init(self.initializer)(layer.key.weight)
            nn.init.zeros_(layer.key.bias)
            apply_init(self.initializer)(layer.value.weight)
            nn.init.zeros_(layer.value.bias)

    def forward(self, rep_2d, rep_3d, num_atoms):
        rep_2d, mask_2d = variadic_to_padded(rep_2d, num_atoms)
        rep_3d, _ = variadic_to_padded(rep_3d, num_atoms)

        x_2d = self.mha_2d_3d(rep_2d, rep_3d, mask_2d)
        x_2d = padded_to_variadic(x_2d, num_atoms)

        x_3d = self.mha_3d_2d(rep_3d, rep_2d, mask_2d)
        x_3d = padded_to_variadic(x_3d, num_atoms)

        return self.layers_2d_3d(x_2d), self.layers_3d_2d(x_3d)
        return x_2d, x_3d


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
