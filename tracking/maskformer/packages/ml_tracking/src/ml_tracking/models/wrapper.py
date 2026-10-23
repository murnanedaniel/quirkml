from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import Tensor, nn
from torch._functorch import config as functorch_config  # noqa: PLC2701
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


def _wsd_lr_lambda(
    total_steps: int,
    initial_lr: float,
    max_lr: float,
    end_lr: float,
    warmup_frac: float,
    cooldown_frac: float,
):
    """Return a WSD learning-rate multiplier for an optimizer based at ``initial_lr``."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if initial_lr <= 0:
        raise ValueError("initial learning rate must be positive")
    if not 0 <= warmup_frac <= 1 or not 0 <= cooldown_frac <= 1 or warmup_frac + cooldown_frac > 1:
        raise ValueError("warmup_frac and cooldown_frac must be in [0, 1] and sum to at most 1")

    max_multiplier = max_lr / initial_lr
    end_multiplier = end_lr / initial_lr
    cooldown_start = 1.0 - cooldown_frac

    def lr_lambda(step: int) -> float:
        progress = min(max(step / total_steps, 0.0), 1.0)
        if warmup_frac > 0 and progress < warmup_frac:
            return 1.0 + (max_multiplier - 1.0) * progress / warmup_frac
        if cooldown_frac > 0 and progress > cooldown_start:
            cooldown_progress = (progress - cooldown_start) / cooldown_frac
            return max_multiplier + (end_multiplier - max_multiplier) * cooldown_progress
        return max_multiplier

    return lr_lambda


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False
            # MTL does not currently support intermediate losses
            assert all(task.has_intermediate_loss is False for task in self.model.tasks)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.predict(outputs)

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str | None = None) -> Tensor:
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        for layer_name, layer_losses in losses.items():
            layer_loss = 0
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)
                    total_loss += loss_value
                    layer_loss += loss_value

            # Log the total loss from the layer
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        # Log the total loss
        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # Log any task specific metrics
        for layer_name in preds:
            # Determine which task list to use based on layer name
            tasks = self.model.encoder_tasks if layer_name == "encoder" else self.model.tasks
            for task in tasks:
                if task.name not in preds[layer_name]:
                    continue

                task_metrics = task.metrics(preds[layer_name][task.name], targets)
                if task_metrics:
                    self.log_dict({f"{stage}/{layer_name}_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        self.log_task_metrics(preds, targets, stage)

        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]], batch_idx: int) -> dict[str, Tensor] | None:
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        outputs, targets, losses = self.model.loss(outputs, targets)

        # Get the predictions from the model, avoid calling predict if possible
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None
        total_loss = self.aggregate_losses(losses, stage="train")

        return {"loss": total_loss}

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss}

    def test_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        outputs, targets, losses = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self) -> None:
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            total_steps = self.trainer.estimated_stepping_batches
            schedule = self.lrs_config.get("schedule", "onecycle").lower()
            if schedule == "onecycle":
                # Keep the legacy/default scheduler behavior exactly unchanged.
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=self.lrs_config["max"],
                    total_steps=total_steps,
                    div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                    final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                    pct_start=float(self.lrs_config["pct_start"]),
                )
            elif schedule == "wsd":
                # For a cooldown branch run, resume a stable-phase checkpoint with
                # --trainer.max_steps=current_step+desired_cooldown_steps and set
                # cooldown_frac=desired_cooldown_steps/(current_step+desired_cooldown_steps).
                sch = torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    _wsd_lr_lambda(
                        total_steps=total_steps,
                        initial_lr=self.lrs_config["initial"],
                        max_lr=self.lrs_config["max"],
                        end_lr=self.lrs_config["end"],
                        warmup_frac=float(self.lrs_config.get("warmup_frac", 0.015)),
                        cooldown_frac=float(self.lrs_config.get("cooldown_frac", 0.15)),
                    ),
                )
            else:
                raise ValueError(f"Unknown learning rate schedule: {schedule}")
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]

        print("Skipping learning rate scheduler.")
        return opt

    def mlt_opt(self, losses: dict[str, Tensor], outputs: dict[str, Tensor]) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        # TODO: Make this not hard coded?
        feature_names = ["query_embed", "key_embed"]

        # Remove any duplicate features that are used by multiple tasks
        features = [outputs["final"][feature_name] for feature_name in feature_names]

        # TODO: Figure out if we can set retain_graph to false somehow, since it uses a lot of memory
        task_losses = [sum(losses["final"][task.name].values()) for task in self.model.tasks]
        mtl_backward(losses=task_losses, features=features, aggregator=UPGrad(), retain_graph=True)

        # Manually perform the optimizer step
        opt.step()
