"""FlexGeo attention for the quirk MaskFormer and hit filter at mu 60 (graph-free).

Ported from the ML-tracking-review thread (worktree pu0-round2, packages/ml_tracking/src/ml_tracking/experiments/
colliderml: geometric_flex.py, rotary_window.py and the HIT_COORDINATES / CheckpointLayer pieces of graph_attention.py;
paper_model.FLEXGEO_FILTER). Their package is also named ml_tracking and imports the pinned hepattn, so the functions are
copied here against this fork's Attention / Encoder (same lineage; _prepare_qkv takes q, k, v here). Logic unchanged:

Every hit attends to hits with |dphi| < dphi and |deta| < deta (FlexAttention block-sparse mask). Hits are sorted by
(eta band, phi) and padded to whole 128-token blocks; q and k carry phi/eta rotary encodings. One unpadded event per
forward (batch 1, the recipe's 1 event/update).
"""

import math
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint

from ml_tracking.models.attention import Attention
from ml_tracking.models.hitfilter import HitFilter
from ml_tracking.models.maskformer import MaskFormer

HIT_COORDINATES = ContextVar("quirk_hit_coordinates", default=None)
BLOCK = 128
PAD_ETA = 1e3  # padding tokens: never inside any eta cut
PHI_FREQUENCIES = (1, 3, 8, 22, 60, 160, 440, 1200)
ETA_FREQUENCIES = tuple(round(10 ** (i * math.log10(300) / 7), 3) for i in range(8))
FLEXGEO = {"dphi": 0.15, "deta": 0.3, "bands": 32, "rotary": True}  # review thread's FlexGeo filter / tracker cuts

compiled_flex = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)  # noqa: SLF001


def rotary_angles(coordinates, phi_freqs=PHI_FREQUENCIES, eta_freqs=ETA_FREQUENCIES):
    """Per-hit rotation angles [hits, pairs] from (r, phi, z) [1, hits, 3] (half phi, half eta pairs)."""
    r, phi, z = coordinates[0].float().unbind(-1)
    eta = torch.asinh(z / r.clamp_min(1e-8))
    return torch.cat([phi[:, None] * phi.new_tensor(phi_freqs), eta[:, None] * eta.new_tensor(eta_freqs)], dim=-1)


def capped_frequencies(n, top):
    """n geometrically spaced rotary frequencies from 1 to top (the defaults reach 1200 in phi, 300 in eta)."""
    return tuple(round(10 ** (i * math.log10(top) / (n - 1)), 3) for i in range(n))


def padded_length(hits):
    """Next multiple of max(BLOCK, 2**(floor(log2(hits)) - 3)) at or above hits."""
    step = max(BLOCK, 2 ** max(int(math.log2(max(hits, 1))) - 3, 0))
    return -(-hits // step) * step


def geometric_order(phi, eta, bands):
    band = ((eta + 4.0) / 8.0 * bands).clamp(0, bands - 1).long()
    return torch.argsort(band * 16.0 + (phi + math.pi) / (2 * math.pi) * 8.0)


def geometric_block_mask(phi, eta, dphi, deta):
    """phi, eta in sorted order, padded to whole blocks (padding eta = PAD_ETA); phi is periodic."""
    n = phi.shape[0]
    if n % BLOCK:
        raise ValueError("Pad the sequence to whole blocks.")
    blocks = n // BLOCK
    phi_b, eta_b = phi.view(blocks, BLOCK), eta.clamp(-10, 10).view(blocks, BLOCK)
    centre = torch.atan2(phi_b.sin().mean(1), phi_b.cos().mean(1))
    half = (torch.remainder(phi_b - centre[:, None] + math.pi, 2 * math.pi) - math.pi).abs().amax(1)
    eta_lo, eta_hi = eta_b.amin(1), eta_b.amax(1)
    dcentre = (torch.remainder(centre[:, None] - centre[None] + math.pi, 2 * math.pi) - math.pi).abs()
    active = (dcentre <= half[:, None] + half[None] + dphi) \
        & (eta_lo[:, None] <= eta_hi[None] + deta) & (eta_lo[None] <= eta_hi[:, None] + deta)
    counts = active.sum(1).int()
    indices = torch.argsort((~active).int(), dim=1, stable=True).int()

    def mask_mod(b, h, q, k):
        d = (phi[q] - phi[k]).abs()
        d = torch.minimum(d, 2 * math.pi - d)
        return (d < dphi) & ((eta[q] - eta[k]).abs() < deta)

    return BlockMask.from_kv_blocks(counts[None, None], indices[None, None], BLOCK_SIZE=BLOCK, mask_mod=mask_mod,
                                    seq_lengths=(n, n))


def rotate_heads_first(x, cos, sin):
    """Rotary rotation for [batch, heads, seq, head_dim]; cos/sin [seq, head_dim/2]."""
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)


class GeometricFlexAttention(Attention):
    @classmethod
    def from_upstream(cls, original):
        original.__class__ = cls
        return original

    def forward(self, q, k=None, v=None, block_mask=None, rotary=None, initial_values=None, kv_mask=None, **kwargs):
        if k is not None or kv_mask is not None or block_mask is None:
            raise ValueError("Geometric flex attention is unmasked self-attention with a block mask.")
        q, k, v = self._prepare_qkv(q, q, q, initial_values)
        if self.attn_type in ("flash", "flash-varlen"):  # B S H D -> B H S D
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if rotary is not None:
            q, k = rotate_heads_first(q, *rotary), rotate_heads_first(k, *rotary)
        out = compiled_flex(q, k, v, block_mask=block_mask)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class CheckpointLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.layer, x, use_reentrant=False, **kwargs)
        return self.layer(x, **kwargs)


class GeometricFlexEncoder(nn.Module):
    """Runs an Encoder's layers over the geometric neighbourhood (coordinates from HIT_COORDINATES)."""

    def __init__(self, encoder, dphi=0.15, deta=0.3, bands=32, rotary=True, checkpoint_layers=True,
                 phi_freq_max=None, eta_freq_max=None):
        """phi_freq_max / eta_freq_max: optional caps on the rotary frequencies (8 per angle, geometric from 1);
        None keeps PHI_FREQUENCIES / ETA_FREQUENCIES."""
        super().__init__()
        self.phi_freqs = PHI_FREQUENCIES if phi_freq_max is None else capped_frequencies(len(PHI_FREQUENCIES), phi_freq_max)
        self.eta_freqs = ETA_FREQUENCIES if eta_freq_max is None else capped_frequencies(len(ETA_FREQUENCIES), eta_freq_max)
        if min(dphi, deta) <= 0 or int(bands) != bands or bands < 1:
            raise ValueError("Require positive cuts and a positive integer number of eta bands.")
        self.attn_type = "flex"
        self.dphi, self.deta, self.bands, self.rotary = float(dphi), float(deta), int(bands), bool(rotary)
        layers = []
        for layer in encoder.layers:
            if rotary and layer.attn.fn.head_dim != 2 * (len(PHI_FREQUENCIES) + len(ETA_FREQUENCIES)):
                raise ValueError("Head width must equal twice the number of rotary frequencies (32).")
            layer.attn.fn = GeometricFlexAttention.from_upstream(layer.attn.fn)
            layers.append(CheckpointLayer(layer) if checkpoint_layers else layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, x_sort_value=None, kv_mask=None, **kwargs):
        coordinates = HIT_COORDINATES.get()
        if x.shape[0] != 1 or coordinates is None or coordinates.shape[1] != x.shape[1] or \
                (kv_mask is not None and not bool(kv_mask.all())):
            raise ValueError("Geometric flex attention requires one unpadded event with hit coordinates.")
        r, phi, z = coordinates[0].float().unbind(-1)
        eta = torch.asinh(z / r.clamp_min(1e-8))
        order = geometric_order(phi, eta, self.bands)
        hits = x.shape[1]
        pad = padded_length(hits) - hits
        phi_s = torch.cat([phi[order], phi.new_zeros(pad)]).contiguous()
        eta_s = torch.cat([eta[order], eta.new_full((pad,), PAD_ETA)]).contiguous()
        mask = geometric_block_mask(phi_s, eta_s, self.dphi, self.deta)
        rotary = None
        if self.rotary:
            angles = rotary_angles(coordinates[:, order], self.phi_freqs, self.eta_freqs)
            angles = torch.cat([angles, angles.new_zeros(pad, angles.shape[1])])
            rotary = (angles.cos(), angles.sin())
        h = torch.cat([x[:, order], x.new_zeros(1, pad, x.shape[2])], dim=1)
        for layer in self.layers:
            h = layer(h, block_mask=mask, rotary=rotary)
        return h[:, :hits][:, torch.argsort(order)]


def _coordinates(inputs):
    return torch.stack([inputs[f"sihit_{f}"] for f in ("r", "phi", "z")], dim=-1)


class FlexMaskFormer(MaskFormer):
    """MaskFormer whose encoder is wrapped in GeometricFlexEncoder(flex); the hit coordinates come from the inputs."""

    def __init__(self, *args, flex: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = GeometricFlexEncoder(self.encoder, **{**FLEXGEO, **(flex or {})})

    def forward(self, inputs):
        token = HIT_COORDINATES.set(_coordinates(inputs))
        try:
            return super().forward(inputs)
        finally:
            HIT_COORDINATES.reset(token)


class FlexHitFilter(HitFilter):
    def __init__(self, *args, flex: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = GeometricFlexEncoder(self.encoder, **{**FLEXGEO, **(flex or {})})

    def forward(self, inputs):
        token = HIT_COORDINATES.set(_coordinates(inputs))
        try:
            return super().forward(inputs)
        finally:
            HIT_COORDINATES.reset(token)
