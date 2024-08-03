import pdb
import torch.nn as nn
from .encoders import AtomEncoder, BondEncoder
from .self_attention import SelfAttentionBlock
from runners.util import (
    apply_init,
    activation_dict,
    padded_to_variadic,
    variadic_to_padded,
)


class TransformerLayer(nn.Module):
    """
    Transformer layer from `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        ff_hidden_dim (int): hidden dimension of feedforward layer
        dropout (float, optional): dropout ratio of attention maps
    """

    def __init__(
        self,
        emb_dim,
        ff_hidden_dim,
        num_heads=4,
        normalizer=nn.Identity(),
        activation="GELU",
        initializer="glorot_normal",
        dropout=0.0,
    ):
        super(TransformerLayer, self).__init__()
        self.mha = SelfAttentionBlock(emb_dim, num_heads, dropout=dropout)
        self.normalizer = normalizer
        self.initializer = initializer

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        act = activation_dict[activation]

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_dim),
            act(),
            nn.Linear(ff_hidden_dim, emb_dim),
        )

    def forward(self, x, mask, batch):
        num_edges = batch.batch[batch.edge_index[0]].bincount()
        pdb.set_trace()
        edge_index, edge_mask = variadic_to_padded(
            batch.edge_index.view(-1, 2), num_edges
        )
        edge_attr, edge_attr_mask = variadic_to_padded(
            batch.edge_attr, num_edges, value=6
        )
        num_atoms = batch.batch.bincount()
        x = padded_to_variadic(x, num_atoms)
        edge_attr = self.bond_encoder(edge_attr)
        x = self.mha(x, x, mask=mask)
        x = self.ff(x)

        return x

    def reset_parameters(self):
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                apply_init(self.initializer)(layer.weight)
                nn.init.zeros_(layer.bias)

        apply_init(self.initializer)(self.mha.query.weight)
        nn.init.zeros_(self.mha.query.bias)
        apply_init(self.initializer)(self.mha.key.weight)
        nn.init.zeros_(self.mha.key.bias)
        apply_init(self.initializer)(self.mha.value.weight)
        nn.init.zeros_(self.mha.value.bias)
