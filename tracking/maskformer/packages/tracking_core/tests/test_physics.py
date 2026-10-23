"""Unit tests for tracking_core.physics — pure-python, no network, no data."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from tracking_core.physics import inv_pt, p_magnitude, qop_to_pt, theta_to_eta


def test_qop_to_pt_at_central():
    pt = qop_to_pt(torch.tensor(0.5), torch.tensor(math.pi / 2))
    assert torch.allclose(pt, torch.tensor(2.0), atol=1e-6)


def test_qop_to_pt_negative_charge():
    pt = qop_to_pt(torch.tensor(-0.5), torch.tensor(math.pi / 2))
    assert torch.allclose(pt, torch.tensor(2.0), atol=1e-6)


def test_theta_to_eta_at_central():
    eta = theta_to_eta(torch.tensor(math.pi / 2))
    assert torch.allclose(eta, torch.tensor(0.0), atol=1e-6)


def test_theta_to_eta_forward_region():
    expected = -math.log(math.tan(math.pi / 8))
    eta = theta_to_eta(torch.tensor(math.pi / 4))
    assert torch.allclose(eta, torch.tensor(expected), atol=1e-6)


def test_numpy_torch_agree_pt():
    theta = math.pi / 3
    qop = 0.25
    np_pt = qop_to_pt(np.array([qop]), np.array([theta]))
    t_pt = qop_to_pt(torch.tensor([qop]), torch.tensor([theta]))
    np.testing.assert_allclose(np_pt, t_pt.numpy(), atol=1e-6)


def test_pt_and_inv_pt_are_reciprocal():
    qop = torch.tensor([0.1, 0.2, 0.3])
    theta = torch.tensor([math.pi / 4, math.pi / 3, math.pi / 2])
    pt = qop_to_pt(qop, theta)
    inv = inv_pt(qop, theta)
    assert torch.allclose(pt * inv, torch.ones_like(pt), atol=1e-5)


def test_p_magnitude():
    qop = torch.tensor([0.5, -0.25])
    p = p_magnitude(qop)
    assert torch.allclose(p, torch.tensor([2.0, 4.0]), atol=1e-6)


@pytest.mark.parametrize("theta_deg", [30.0, 45.0, 60.0, 90.0, 120.0])
def test_pt_scales_with_sin_theta(theta_deg):
    qop = torch.tensor(0.1)
    theta = torch.tensor(math.radians(theta_deg))
    pt = qop_to_pt(qop, theta)
    expected = 10.0 * math.sin(math.radians(theta_deg))
    assert torch.allclose(pt, torch.tensor(expected), atol=1e-5)
