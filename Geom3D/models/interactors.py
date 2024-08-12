import pdb
import torch
import torch.nn as nn
from .self_attention import SelfAttentionBlock

from runners.util import (
    apply_init,
    activation_dict,
    variadic_to_padded,
    padded_to_variadic,
)


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

        self.mha = SelfAttentionBlock(256, 1, dropout=0.0)

        self.ln_2d = nn.LayerNorm(emb_dim)
        self.ln_3d = nn.LayerNorm(emb_dim)

        self.down_projector = nn.Linear(emb_dim, 256)
        self.up_projector = nn.Linear(256, emb_dim)

        for _ in range(1):
            layers_2d_3d.append(nn.Linear(emb_dim, emb_dim))
            layers_2d_3d.append(act())

            layers_3d_2d.append(nn.Linear(emb_dim, emb_dim))
            layers_3d_2d.append(act())

            # layers_2d_3d.append(nn.LayerNorm(emb_dim))
            # layers_3d_2d.append(nn.LayerNorm(emb_dim))

        self.layers_2d_3d = nn.Sequential(*layers_2d_3d)
        self.layers_3d_2d = nn.Sequential(*layers_3d_2d)

    def reset_parameters(self):
        for layer in [
            self.layers_2d_3d,
            self.layers_3d_2d,
            self.down_projector,
            self.up_projector,
        ]:
            if isinstance(layer, nn.Linear):
                apply_init(self.initializer)(layer.weight)
                nn.init.zeros_(layer.bias)

        apply_init(self.initializer)(self.mha.query.weight)
        nn.init.zeros_(self.mha.query.bias)
        apply_init(self.initializer)(self.mha.key.weight)
        nn.init.zeros_(self.mha.key.bias)
        apply_init(self.initializer)(self.mha.value.weight)
        nn.init.zeros_(self.mha.value.bias)

    def forward(self, rep_2d, rep_3d, num_atoms):
        rep_2d = self.ln_2d(rep_2d)
        rep_3d = self.ln_3d(rep_3d)

        # orig_2d, orig_3d = rep_2d, rep_3d
        rep_2d = self.down_projector(rep_2d)
        rep_3d = self.down_projector(rep_3d)

        rep_2d, mask_2d = variadic_to_padded(rep_2d, num_atoms)
        rep_3d, _ = variadic_to_padded(rep_3d, num_atoms)

        rep = torch.cat([rep_2d, rep_3d], dim=1)
        mask = torch.cat([mask_2d, mask_2d], dim=1)

        output = self.mha(rep, rep, mask)

        x_2d, x_3d = torch.split(output, output.shape[1] // 2, dim=1)

        x_2d = padded_to_variadic(x_2d, num_atoms)
        x_3d = padded_to_variadic(x_3d, num_atoms)

        x_2d = self.up_projector(x_2d)
        x_3d = self.up_projector(x_3d)

        return self.layers_2d_3d(x_2d), self.layers_3d_2d(x_3d)
        # return orig_2d * self.layers_2d_3d(x_2d), orig_3d * self.layers_3d_2d(x_3d)


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


class InteractionOuterProd(nn.Module):
    def __init__(
        self,
        emb_dim,
        downsampled_dim,
        dropout=0.0,
        num_layers=1,
        normalizer=nn.Identity,
        initializer="glorot",
        activation="GELU",
    ):
        super(InteractionOuterProd, self).__init__()
        self.emb_dim = emb_dim
        self.downsampled_dim = downsampled_dim
        self.dropout = dropout
        self.num_layers = num_layers

        act = activation_dict[activation]
        self.initializer = initializer

        self.downsampler_2d_3d = nn.Linear(emb_dim, downsampled_dim)
        self.downsampler_3d_2d = nn.Linear(emb_dim, downsampled_dim)

        # self.layers_2d_3d = self.create_mlp(
        #     num_layers, downsampled_dim, act, nn.LayerNorm, dropout, pre_norm=True
        # )
        # self.layers_3d_2d = self.create_mlp(
        #     num_layers, downsampled_dim, act, nn.LayerNorm, dropout, pre_norm=True
        # )
        self.layers_2d_3d = nn.Linear(downsampled_dim, downsampled_dim)
        self.layers_3d_2d = nn.Linear(downsampled_dim, downsampled_dim)

        self.processing_layers = self.create_mlp(
            2, 2 * downsampled_dim, act, nn.Identity, dropout, pre_norm=True
        )

        self.upsampler_2d_3d = nn.Linear(downsampled_dim, emb_dim)
        self.upsampling_layers_2d_3d = self.create_mlp(
            num_layers, emb_dim, act, nn.LayerNorm, dropout, pre_norm=True
        )

        self.upsampler_3d_2d = nn.Linear(downsampled_dim, emb_dim)
        self.upsampling_layers_3d_2d = self.create_mlp(
            num_layers, emb_dim, act, nn.LayerNorm, dropout, pre_norm=True
        )

        self.row_weights_2d_3d = nn.Sequential(
            *[
                # self.create_mlp(
                #     1,
                #     downsampled_dim,
                #     act,
                #     nn.LayerNorm,
                #     dropout,
                #     pre_norm=True,
                # ),
                nn.Linear(downsampled_dim, 1),
            ]
        )
        self.row_weights_3d_2d = nn.Sequential(
            *[
                # self.create_mlp(
                #     1,
                #     downsampled_dim,
                #     act,
                #     nn.LayerNorm,
                #     dropout,
                #     pre_norm=True,
                # ),
                nn.Linear(downsampled_dim, 1),
            ]
        )

        self.ln_2d = nn.LayerNorm(downsampled_dim)
        self.ln_3d = nn.LayerNorm(downsampled_dim)

    def create_mlp(self, num_layers, emb_dim, act, normalizer, dropout, pre_norm=True):
        layers = []
        for _ in range(num_layers):
            if pre_norm:
                layers.append(normalizer(emb_dim))
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            if not pre_norm:
                layers.append(normalizer(emb_dim))

        return nn.Sequential(*layers)

    def reset_parameters(self):
        all_layers = [
            self.downsampler_2d_3d,
            self.downsampler_3d_2d,
            self.layers_2d_3d,
            self.layers_3d_2d,
            self.upsampler_2d_3d,
            self.upsampler_3d_2d,
            self.row_weights_2d_3d,
            self.row_weights_3d_2d,
        ]
        for layer in all_layers:
            if isinstance(layer, nn.Linear):
                apply_init(self.initializer)(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, rep_2d, rep_3d, num_atoms
    ):  # rep_2d: (N, emb_dim), rep_3d: (N, emb_dim)
        down_rep_2d = self.downsampler_2d_3d(rep_2d)
        down_rep_3d = self.downsampler_2d_3d(rep_3d)

        # down_rep_2d = self.layers_2d_3d(down_rep_2d)
        # down_rep_3d = self.layers_3d_2d(down_rep_3d)

        # down_rep_2d = rep_2d
        # down_rep_3d = rep_3d

        x = down_rep_2d.unsqueeze(-1) * down_rep_3d.unsqueeze(-2)
        # x = x.flatten(-2, -1)

        # x = self.middle_layers(x)

        # x = x.view(x.shape[0], self.downsampled_dim, self.downsampled_dim)

        x = self.ln_2d(x)

        x_2d = self.row_weights_2d_3d(x).squeeze(-1)
        x_3d = self.row_weights_2d_3d(x.transpose(-1, -2)).squeeze(-1)
        # x_2d, x_3d = x.mean(dim=-1), x.mean(dim=-2)

        # x_2d_ = self.layers_2d_3d(x_2d)
        # x_3d_ = self.layers_2d_3d(x_3d)

        # cat
        catted = torch.cat([x_2d, x_3d], dim=-1)
        x = self.processing_layers(catted)

        x_2d, x_3d = torch.split(x, x.shape[-1] // 2, dim=-1)

        # x_2d = x_2d + x_2d_
        # x_3d = x_3d + x_3d_

        # use the same networks for both 3d and 2d
        x_2d, x_3d = self.upsampler_2d_3d(x_2d), self.upsampler_2d_3d(x_3d)

        # return rep_2d * x_2d, rep_3d * x_3d
        return x_2d, x_3d

        # x = self.ln(x)

        # x, mask_2d = variadic_to_padded(x, num_atoms)
        # x = self.mha(x, x, mask=mask_2d)
        # x = padded_to_variadic(x, num_atoms)

        # x = self.upsampler(x)
        # x = self.upsampling_layers(x)

        return rep_2d * x, rep_3d * x

        # x_2d = rep_2d * x
        # x_3d = rep_3d * x

        # return x_2d, x_3d
