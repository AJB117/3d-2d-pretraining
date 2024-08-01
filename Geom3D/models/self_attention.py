import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        dropout (float, optional): dropout ratio of attention maps
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads)
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, src, tgt, mask):
        """
        Perform self attention over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        query = self.query(src).transpose(0, 1)
        key = self.key(tgt).transpose(0, 1)
        value = self.value(tgt).transpose(0, 1)

        if mask is not None:
            mask = (~mask.bool()).squeeze(-1)

        output = self.attn(query, key, value, key_padding_mask=mask)[0].transpose(0, 1)

        return output
