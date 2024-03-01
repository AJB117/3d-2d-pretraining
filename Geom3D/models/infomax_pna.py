import tqdm
import pdb
from typing import Dict, List, Union, Callable

import torch
import numpy as np
from functools import partial

from torch import nn

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing, DegreeScalerAggregation
from torch_geometric.utils import degree
from torch import Tensor
from typing import Optional
from torch_scatter import scatter
from .molecule_gnn_model import MLP
from torch.utils.data import DataLoader


def global_min_pool(
    x: Tensor, batch: Optional[Tensor], size: Optional[int] = None
) -> Tensor:
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.min(dim=dim, keepdim=x.dim() <= 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce="min")


EPS = 1e-5


def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNA(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(
        self,
        hidden_dim,
        aggregators: List[str],
        scalers: List[str],
        readout_aggregators: List[str],
        readout_hidden_dim=None,
        residual: bool = True,
        pairwise_distances: bool = False,
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        propagation_depth: int = 5,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        batch_norm_momentum=0.1,
        avg_d={"log": 1.0},
        **kwargs,
    ):
        super(PNA, self).__init__()
        self.node_gnn = PNAGNN(
            hidden_dim=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            residual=residual,
            pairwise_distances=pairwise_distances,
            activation=activation,
            last_activation=last_activation,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            propagation_depth=propagation_depth,
            dropout=dropout,
            posttrans_layers=posttrans_layers,
            pretrans_layers=pretrans_layers,
            batch_norm_momentum=batch_norm_momentum,
            avg_d=avg_d,
        )
        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        # self.output = MLP(
        #     in_dim=hidden_dim * len(self.readout_aggregators),
        #     hidden_size=readout_hidden_dim,
        #     mid_batch_norm=readout_batchnorm,
        #     out_dim=target_dim,
        #     layers=readout_layers,
        #     batch_norm_momentum=batch_norm_momentum,
        # )

    @staticmethod
    def get_degree_histogram(loader: DataLoader) -> Tensor:
        r"""Returns the degree histogram to be used as input for the :obj:`deg`
        argument in :class:`PNAConv`.
        """
        print("getting histogram...")
        try:
            deg_histogram = np.load(f"deg_histogram_{str(loader.dataset)}.npy")
            print("histogram loaded")
            return torch.from_numpy(deg_histogram)
        except:
            deg_histogram = torch.zeros(1, dtype=torch.long)
            for data in tqdm.tqdm(loader):
                deg = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg_bincount = torch.bincount(deg, minlength=deg_histogram.numel())
                deg_histogram = deg_histogram.to(deg_bincount.device)
                if deg_bincount.numel() > deg_histogram.numel():
                    deg_bincount[: deg_histogram.size(0)] += deg_histogram
                    deg_histogram = deg_bincount
                else:
                    assert deg_bincount.numel() == deg_histogram.numel()
                    deg_histogram += deg_bincount

            print("histogram done")
            # save histogram
            np.save(
                f"deg_histogram_{str(loader.dataset)}.npy", deg_histogram.cpu().numpy()
            )

        return deg_histogram

    def forward(self, x, edge_index, edge_attr):
        h = self.node_gnn(x, edge_index, edge_attr)
        return h
        # graph_representations = []
        # for agg in self.readout_aggregators:
        #     if agg == "min":
        #         graph_representations.append(global_min_pool(h, batch))
        #     elif agg == "max":
        #         graph_representations.append(global_max_pool(h, batch))
        #     elif agg == "mean":
        #         graph_representations.append(global_mean_pool(h, batch))
        #     elif agg == "sum":
        #         graph_representations.append(global_add_pool(h, batch))
        #     else:
        #         raise ValueError("Unknown readout aggregator {}".format(agg))

        # readout = torch.cat(graph_representations, dim=-1)
        # return self.output(readout)


class PNAGNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        aggregators: List[str],
        scalers: List[str],
        residual: bool = True,
        pairwise_distances: bool = False,
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum=0.1,
        propagation_depth: int = 5,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        avg_d={"log": 1.0},
        **kwargs,
    ):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(
                    in_dim=hidden_dim,
                    out_dim=int(hidden_dim),
                    in_dim_edges=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    pairwise_distances=pairwise_distances,
                    residual=residual,
                    dropout=dropout,
                    activation=activation,
                    last_activation=last_activation,
                    mid_batch_norm=mid_batch_norm,
                    last_batch_norm=last_batch_norm,
                    avg_d=avg_d,
                    posttrans_layers=posttrans_layers,
                    pretrans_layers=pretrans_layers,
                    batch_norm_momentum=batch_norm_momentum,
                ),
            )
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        h_in = x
        for mp_layer in self.mp_layers:
            h = mp_layer(x, edge_index, edge_attr)
            if self.residual:
                h = h + h_in
                h_in = h

        return h


class PNALayer(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        aggregators: List[str],
        scalers: List[str],
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        dropout: float = 0.0,
        residual: bool = True,
        pairwise_distances: bool = False,
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum=0.1,
        avg_d: Dict[str, float] = {"log": 1.0},
        posttrans_layers: int = 2,
        pretrans_layers: int = 1,
    ):
        aggr = DegreeScalerAggregation(aggregators, scalers, avg_d)
        super(PNALayer, self).__init__(aggr=aggr, node_dim=0)

        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1)
            if self.pairwise_distances
            else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_layers,
            mid_activation=activation,
            dropout=dropout,
            last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.posttrans = MLP(
            in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            hidden_size=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward(self, x, edge_index, edge_attr):
        # pretransformation on edges
        h = x
        h_in = h
        # edge_attr = self.pretrans_edges(x, edge_index, edge_attr)
        # pdb.set_trace()

        # aggregation
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        h = torch.cat([h, h_in], dim=-1)
        # post-transformation
        h = self.posttrans(h)

        return h

    def message(self, x_i, x_j, edge_attr):
        r"""
        The message function to generate messages along the edges.
        """
        z2 = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.pretrans(z2)

    # def aggregate(self, inputs, index, dim_size=None):
    #     h_to_cat = [aggr(h=inputs) for aggr in self.aggregators]
    #     h = torch.cat(h_to_cat, dim=-1)

    #     pdb.set_trace()
    #     if len(self.scalers) > 1:
    #         h = torch.cat(
    #             [scale(h, D=index, avg_d=self.avg_d) for scale in self.scalers],
    #             dim=-1,
    #         )

    #     return h

    def pretrans_edges(self, x, edge_index, edge_attr) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        h_src = x[edge_index[0]]
        h_dst = x[edge_index[1]]
        z2 = torch.cat(
            [h_src, h_dst, edge_attr],
            dim=-1,
        )
        return self.pretrans(z2)
