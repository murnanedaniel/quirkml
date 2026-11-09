"""Degree-aware variants of the InteractionGNN (gnn/common.py) for mu 60, where pair-edge recall falls with hit degree
(0.69 below 50 graph edges, 0.23 at 100-200). agg = "sum" (the original), "mean" (in/out aggregates divided by the
in/out degree), "summean" (sum and mean aggregates plus log1p(degree) as node-update inputs)."""
import torch
import torch.nn as nn

from common import InteractionGNN, mlp


class DegreeGNN(InteractionGNN):
    def __init__(self, n_in, n_edge_in, hidden=128, n_iter=8, n_layers=2, agg="mean"):
        super().__init__(n_in=n_in, n_edge_in=n_edge_in, hidden=hidden, n_iter=n_iter, n_layers=n_layers)
        self.agg = agg
        if agg == "summean":
            self.node_net = mlp(5 * hidden + 2, hidden, hidden, n_layers)

    def forward(self, f, src, dst, efeat):
        h = self.node_enc(f)
        e = self.edge_enc(torch.cat([h[src], h[dst], efeat], 1))
        n = h.shape[0]
        one = torch.ones(len(src), device=h.device, dtype=torch.float32)
        din = torch.zeros(n, device=h.device).index_add_(0, dst, one).clamp_min(1)[:, None]
        dout = torch.zeros(n, device=h.device).index_add_(0, src, one).clamp_min(1)[:, None]
        for _ in range(self.n_iter):
            agg_in = torch.zeros_like(h).index_add_(0, dst, e)
            agg_out = torch.zeros_like(h).index_add_(0, src, e)
            if self.agg == "mean":
                x = torch.cat([h, agg_in / din.to(h.dtype), agg_out / dout.to(h.dtype)], 1)
            else:
                x = torch.cat([h, agg_in, agg_out, agg_in / din.to(h.dtype), agg_out / dout.to(h.dtype),
                               torch.log1p(din).to(h.dtype), torch.log1p(dout).to(h.dtype)], 1)
            h = h + self.node_net(x)
            e = e + self.edge_net(torch.cat([h[src], h[dst], e], 1))
        return self.out(torch.cat([h[src], h[dst], e], 1)).squeeze(-1)


def patch_build_model(T, agg):
    """Make train_gnn.build_model return the degree-aware model (same inputs as the original)."""
    if agg == "sum":
        return
    from common import DIRECTION, N_DIR_EDGE, N_DIR_NODE, N_FEAT

    def build_model(hidden, n_iter, layers):
        d = DIRECTION["on"]
        return DegreeGNN(n_in=N_FEAT + (N_DIR_NODE if d else 0), n_edge_in=T.N_EDGE_IN[T.CLUSTER] + (N_DIR_EDGE if d else 0),
                         hidden=hidden, n_iter=n_iter, n_layers=layers, agg=agg).to(T.DEV)
    T.build_model = build_model
