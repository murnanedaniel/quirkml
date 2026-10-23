from types import SimpleNamespace

import pytest
import torch

from ml_tracking.models.wrapper import ModelWrapper, _wsd_lr_lambda


def test_wsd_lr_lambda_key_steps():
    lr_lambda = _wsd_lr_lambda(
        total_steps=1000,
        initial_lr=1e-6,
        max_lr=5e-5,
        end_lr=1e-6,
        warmup_frac=0.015,
        cooldown_frac=0.15,
    )

    assert lr_lambda(0) == pytest.approx(1.0)
    assert lr_lambda(15) == pytest.approx(50.0)
    assert lr_lambda(500) == pytest.approx(50.0)
    assert lr_lambda(850) == pytest.approx(50.0)
    assert lr_lambda(1000) == pytest.approx(1.0)


def test_onecycle_remains_default_scheduler():
    wrapper = ModelWrapper(
        name="tiny",
        model=torch.nn.Linear(2, 1),
        lrs_config={
            "initial": 1e-6,
            "max": 5e-5,
            "end": 1e-6,
            "pct_start": 0.1,
            "skip_scheduler": False,
            "weight_decay": 1e-2,
        },
    )
    wrapper._trainer = SimpleNamespace(estimated_stepping_batches=1000)

    _, schedulers = wrapper.configure_optimizers()

    assert isinstance(schedulers[0]["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
    assert schedulers[0]["interval"] == "step"
