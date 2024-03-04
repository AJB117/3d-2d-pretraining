import torch
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


class E_GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        positions_weight=1.0,
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=False,
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.positions_weight = positions_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_positions_nf = 1

        self.bond_encoder = BondEncoder(hidden_nf)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_positions_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        positions_mlp = []
        positions_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        positions_mlp.append(act_fn)
        positions_mlp.append(layer)
        if self.tanh:
            positions_mlp.append(nn.Tanh())
            self.positions_range = nn.Parameter(torch.ones(1)) * 3
        self.positions_mlp = nn.Sequential(*positions_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())
        return

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def positions_model(self, positions, edge_index, positions_diff, edge_feat):
        row, _ = edge_index
        trans = positions_diff * self.positions_mlp(edge_feat)
        trans = torch.clamp(
            trans, min=-100, max=100
        )  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_sum(trans, row, num_segments=positions.size(0))
        positions = positions + agg * self.positions_weight
        return positions

    def positions2radial(self, edge_index, positions):
        row, col = edge_index
        positions_diff = positions[row] - positions[col]
        radial = torch.sum((positions_diff) ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            positions_diff = positions_diff / (norm)
        return radial, positions_diff

    def forward(self, h, positions, edge_index, node_attr=None, edge_attr=None):
        """
        h: (N, emb)
        positions: (N, 3)
        edge_index: (2, M)
        node_attr: None or (N, node_input_dim), where node_input_dim=1
        edge_attr: None or (M, edge_input_dim)
        """
        row, col = edge_index
        radial, positions_diff = self.positions2radial(
            edge_index, positions
        )  # (2N, 1), (N, 3)
        edge_embedding = self.bond_encoder(edge_attr)
        edge_feat = self.edge_model(
            h[row], h[col], radial, edge_attr=edge_embedding
        )  # (M, n_emb)

        positions = self.positions_model(
            positions, edge_index, positions_diff, edge_feat
        )  # (M, 3)
        h, agg = self.node_model(
            h, edge_index, edge_feat, node_attr
        )  # (N, emb_dim), (N, emb_dim*2 + input_node_dim)
        return h, positions, edge_attr

    def forward_with_gathered_index(
        self, h, positions, edge_index, node_attr, periodic_index_mapping
    ):
        row, col = edge_index
        radial, positions_diff = self.positions2radial(
            edge_index, positions
        )  # (2N, 1), (N, 3)

        gathered_row = periodic_index_mapping[row]
        gathered_col = periodic_index_mapping[col]
        edge_feat = self.edge_model(
            h[gathered_row], h[gathered_col], radial, edge_attr
        )  # (M, n_emb)

        h, agg = self.node_model(
            h, edge_index, edge_feat, node_attr
        )  # (N, emb_dim), (N, emb_dim*2 + input_node_dim)
        return h, positions, edge_attr


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        positions_weight=1.0,
        attention=True,
        node_attr=True,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.atom_encoder = AtomEncoder(hidden_nf)

        self.node_attr = node_attr

        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        for i in range(0, n_layers):
            layer_ = E_GCL(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                edges_in_d=in_edge_nf,
                nodes_attr_dim=n_node_attr,
                positions_weight=positions_weight,
                act_fn=act_fn,
                recurrent=True,
                attention=attention,
            )
            self.add_module("gcl_%d" % i, layer_)

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        return

    def forward(self, x, positions, edge_index, edge_attr=None):
        h = self.atom_encoder(x)

        for i in range(self.n_layers):
            node_attr = x if self.node_attr else None
            h, positions, _ = self._modules["gcl_%d" % i](
                h, positions, edge_index, node_attr=node_attr, edge_attr=edge_attr
            )

        h = self.node_dec(h)
        return h, positions

    def forward_with_gathered_index(
        self, gathered_x, positions, edge_index, periodic_index_mapping
    ):
        h = self.atom_encoder(gathered_x)

        for i in range(self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i].forward_with_gathered_index(
                    h,
                    positions,
                    edge_index,
                    node_attr=gathered_x,
                    periodic_index_mapping=periodic_index_mapping,
                )
            # else:
            #     h, _, _ = self._modules["gcl_%d" % i](
            #         h, positions, edge_index, node_attr=None, edge_attr=edge_attr
            #     )

        h = self.node_dec(h)
        return h


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 16
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.rand(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
