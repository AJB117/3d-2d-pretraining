"""
Transformer implementation by leffff (https://github.com/leffff/graphormer-pyg) without the SPD+C embeddings
"""
from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1)

class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij)
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_paths,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)
        a = (a + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_paths,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new

class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x