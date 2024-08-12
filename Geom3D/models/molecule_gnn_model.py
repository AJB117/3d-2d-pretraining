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
    GPSConv,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
from .encoders import AtomEncoder, BondEncoder
from typing import List
from runners.util import apply_init
import math
from typing import List, Optional, Union, Dict, Any
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.nn.resolver import activation_resolver
from torch import Tensor


# the following MLP and GTConv classes credited to https://github.com/pgniewko/gt-pyg
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[int, List[int]],
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dims (Union[int, List[int]]): Hidden layer dimensions.
                If int, same hidden dimension is used for all layers.
            num_hidden_layers (int, optional): Number of hidden layers. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.0.
            act (str, optional): Activation function name. Default is "relu".
            act_kwargs (Dict[str, Any], optional): Additional arguments for the activation function.
                                                   Default is None.
        """
        super(MLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_hidden_layers

        assert len(hidden_dims) == num_hidden_layers

        hidden_dims = [input_dim] + hidden_dims
        layers = []

        for i_dim, o_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(activation_resolver(act, **(act_kwargs or {})))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Any): Input tensor.

        Returns:
            Any: Output tensor.
        """
        return self.mlp(x)


class GTConv(MessagePassing):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        edge_in_dim: Optional[int] = None,
        num_heads: int = 8,
        gate=False,
        qkv_bias=False,
        dropout: float = 0.0,
        norm: str = "bn",
        act: str = "relu",
        aggregators: List[str] = ["sum"],
    ):
        """
        Graph Transformer Convolution (GTConv) module.

        Args:
            node_in_dim (int): Dimensionality of the input node features.
            hidden_dim (int): Dimensionality of the hidden representations.
            edge_in_dim (int, optional): Dimensionality of the input edge features.
                                         Default is None.
            num_heads (int, optional): Number of attention heads. Default is 8.
            dropout (float, optional): Dropout probability. Default is 0.0.
            gate (bool, optional): Use a gate attantion mechanism.
                                   Default is False
            qkv_bias (bool, optional): Bias in the attention mechanism.
                                       Default is False
            norm (str, optional): Normalization type. Options: "bn" (BatchNorm), "ln" (LayerNorm).
                                  Default is "bn".
            act (str, optional): Activation function name. Default is "relu".
            aggregators (List[str], optional): Aggregation methods for the messages aggregation.
                                               Default is ["sum"].
        """
        super().__init__(node_dim=0, aggr=MultiAggregation(aggregators, mode="cat"))

        assert (
            "sum" in aggregators
        )  # makes sure that the original sum_j is always part of the message passing
        assert hidden_dim % num_heads == 0
        assert (edge_in_dim is None) or (edge_in_dim > 0)

        self.aggregators = aggregators
        self.num_aggrs = len(aggregators)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, node_in_dim, bias=True)

        if edge_in_dim is not None:
            self.WE = nn.Linear(edge_in_dim, hidden_dim, bias=True)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)
            self.ffn_e = MLP(
                input_dim=edge_in_dim,
                output_dim=edge_in_dim,
                hidden_dims=hidden_dim,
                num_hidden_layers=1,
                dropout=dropout,
                act=act,
            )
            if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError
        else:
            assert gate is False
            self.WE = self.register_parameter("WE", None)
            self.WOe = self.register_parameter("WOe", None)
            self.ffn_e = self.register_parameter("ffn_e", None)
            self.norm1e = self.register_parameter("norm1e", None)
            self.norm2e = self.register_parameter("norm2e", None)

        if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
            self.norm1 = nn.BatchNorm1d(node_in_dim)
            self.norm2 = nn.BatchNorm1d(node_in_dim)
        elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
            self.norm1 = nn.LayerNorm(node_in_dim)
            self.norm2 = nn.LayerNorm(node_in_dim)

        if gate:
            self.n_gate = nn.Linear(node_in_dim, hidden_dim, bias=True)
            self.e_gate = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        else:
            self.n_gate = self.register_parameter("n_gate", None)
            self.e_gate = self.register_parameter("e_gate", None)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.ffn = MLP(
            input_dim=node_in_dim,
            output_dim=node_in_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )

        self.num_heads = num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias

        self.reset_parameters()

    def reset_parameters(self):
        """
        Note: The output of the Q-K-V layers does not pass through the activation layer (as opposed to the input),
              so the variance estimation should differ by a factor of two from the default
              kaiming_uniform initialization.
        """
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE.weight)
            nn.init.xavier_uniform_(self.WOe.weight)

    def forward(self, x, edge_index, edge_attr=None):
        x_ = x

        if edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)

        edge_attr_ = edge_attr

        Q = self.WQ(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.WK(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.WV(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        if self.gate:
            G = self.n_gate(x).view(
                -1, self.num_heads, self.hidden_dim // self.num_heads
            )
        else:
            G = torch.ones_like(V)  # G*V = V

        out = self.propagate(
            edge_index, Q=Q, K=K, V=V, G=G, edge_attr=edge_attr, size=None
        )
        out = out.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation

        # NODES
        out = self.dropout_layer(out)
        out = self.WO(out) + x_
        out = self.norm1(out)
        # FFN--nodes
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)

        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            # EDGES
            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + edge_attr_  # Residual connection
            out_eij = self.norm1e(out_eij)
            # FFN--edges
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)

        return out

    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j) / math.sqrt(d_k)
        if self.edge_in_dim is not None:
            assert edge_attr is not None
            E = self.WE(edge_attr).view(
                -1, self.num_heads, self.hidden_dim // self.num_heads
            )
            qijk = E * qijk
            self._eij = qijk
        else:
            self._eij = None

        if self.gate:
            assert edge_attr is not None
            e_gate = self.e_gate(edge_attr).view(
                -1, self.num_heads, self.hidden_dim // self.num_heads
            )
            qijk = torch.mul(qijk, torch.sigmoid(e_gate))

        qijk = (Q_i * K_j).sum(dim=-1) / math.sqrt(d_k)

        alpha = softmax(
            qijk, index
        )  # Log-Sum-Exp trick used. No need for clipping (-5,5)

        if self.gate:
            V_j_g = torch.mul(V_j, torch.sigmoid(G_j))
        else:
            V_j_g = V_j

        return alpha.view(-1, self.num_heads, 1) * V_j_g

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}({self.node_in_dim}, "
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate})"
        )


class TransformerConv(MessagePassing):
    def __init__(self, emb_dim, heads=4):
        super(TransformerConv, self).__init__(node_dim=0)
        self.heads = heads
        self.emb_dim = emb_dim

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.bond_encoder = BondEncoder(emb_dim=heads * emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.relu(alpha)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GINBlock(MessagePassing):
    def __init__(self, emb_dim, initializer="glorot", normalizer=nn.Identity()):
        super(GINBlock, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            # normalizer,
            torch.nn.SiLU(),
            # torch.nn.Linear(emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        self.initializer = initializer

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                apply_init(self.initializer)(m.weight)
                zeros(m.bias)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.bond_encoder = BondEncoder(emb_dim=heads * emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN_NoAtom(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN_NoAtom, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "GIN":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "GCN":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "GAT":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "GraphSAGE":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "GIN":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "GCN":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "GAT":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "GraphSAGE":
                self.gnns.append(GraphSAGEConv(emb_dim))
            elif gnn_type == "Transformer":
                self.gnns.append(TransformerConv(emb_dim))
            elif gnn_type == "GT":
                self.gnns.append(
                    GTConv(
                        emb_dim,
                        emb_dim,
                        emb_dim,
                        num_heads=4,
                        gate=False,
                        qkv_bias=True,
                        dropout=drop_ratio,
                        norm="ln",
                        act="relu",
                    )
                )
            elif gnn_type == "GPS":
                self.gnns.append(
                    GPSConv(emb_dim, GINConv(emb_dim), heads=2, act="swish")
                )

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.JK = args.JK
        graph_pooling = args.graph_pooling
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.molecule_model = molecule_model

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)

        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)

        output = self.graph_pred_linear(graph_representation)

        return output


class GraphPredLinear(nn.Module):
    def __init__(self, layer_sizes: List[int], num_tasks: int, graph_pooling: str):
        super(GraphPredLinear, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer_size in layer_sizes:
            in_dim, out_dim = layer_size
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.graph_pred_linear = nn.Linear(out_dim, num_tasks)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, graph_representation, batch):
        graph_representation = self.pool(graph_representation, batch)

        for i, layer in enumerate(self.layers):
            graph_representation = layer(graph_representation)
            graph_representation = self.batch_norms[i](graph_representation)
            graph_representation = F.relu(graph_representation)

        output = self.graph_pred_linear(graph_representation)

        return output
