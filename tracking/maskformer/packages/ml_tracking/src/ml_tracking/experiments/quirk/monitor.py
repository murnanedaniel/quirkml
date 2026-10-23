"""Training-health logging (tuning-playbook quantities): grad norm, steps/s, effective batch, GPU memory."""

from time import perf_counter

import torch
from lightning import Callback


class TrainingMonitor(Callback):
    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.every = log_every_n_steps
        self._t0 = None
        self._step0 = 0

    def on_train_start(self, trainer, pl_module):
        self._t0 = perf_counter()
        self._step0 = trainer.global_step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Called before gradient clipping: this is the raw total L2 norm.
        # Lightning flushes step metrics when (step + 1) % log_every_n_steps == 0.
        if (trainer.global_step + 1) % self.every:
            return
        norms = [p.grad.detach().float().norm(2) for p in pl_module.parameters() if p.grad is not None]
        if norms:
            pl_module.log("train/grad_norm", torch.stack(norms).norm(2), on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every or step == self._step0:
            return
        now = perf_counter()
        sps = (step - self._step0) / max(now - self._t0, 1e-9)
        self._t0, self._step0 = now, step
        dm = trainer.datamodule
        eff = getattr(dm, "batch_size", 1) * trainer.accumulate_grad_batches * trainer.world_size
        metrics = {"train/steps_per_s": sps, "train/effective_batch_size": float(eff), "train/events_per_s": sps * eff}
        if torch.cuda.is_available():
            metrics["train/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        pl_module.log_dict(metrics, on_step=True, on_epoch=False)
