import torch
import torch.nn as nn
from .self_attention import SelfAttentionBlock

from runners.util import apply_init

activation_dict = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU, "Swish": nn.SiLU}


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

        layers = []
        emb_dim = emb_dim * 2 if interaction_agg == "cat" else emb_dim
        for _ in range(2):
            layers.append(SelfAttentionBlock(emb_dim, num_heads, dropout))
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(normalizer(emb_dim))
            layers.append(act())

        self.layers = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.mha.children():
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

        # make mask where 2d can only attend to 3d and vice versa
        # 2d -> 3d
        mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
        mask[: rep_2d.size(0), rep_2d.size(0) :] = float("-inf")
        mask = mask.unsqueeze(0).expand(x.size(0), -1, -1)

        # 3d -> 2d
        mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
        mask[rep_2d.size(0) :, : rep_2d.size(0)] = float("-inf")
        mask = mask.unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.mha(x, x, mask)
        return x


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
