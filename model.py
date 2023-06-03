import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class GNNLayer(nn.Module):
    """
    GNN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(GNNLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        new = semantic_embeddings.squeeze()
        return new

class HoGANet(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HoGANet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            GNNLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                GNNLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)