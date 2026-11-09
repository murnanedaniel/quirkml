"""SpikeGuard ported verbatim from the ML-tracking-review thread (branch guard-backstop-fix, lightning_fresh.py,
commits 2aa62b6 and 352c500): skip an update whose finite pre-clip gradient norm exceeds factor x the median of the
last `window` applied updates; steady-scale backstop; stays armed across resume."""

import math


class SpikeGuard:
    """Median of the last `window` finite pre-clip gradient norms of APPLIED updates; a finite norm above `factor`
    times that median is a spike once `min_history` norms are known (factor 0 disables). Skipped norms are kept apart
    (`skipped`), so a burst cannot poison the median (FlexGeo decay branch 2026-10-23: with skipped norms in the
    median, burst norms of 1e10 to 1e15 took it to 7.8e6 and 8 giant updates went through). After `max_consecutive`
    spikes in a row the guard takes the skipped run's norms as the new scale (history replaced by them) and applies
    the update: a lasting change of scale costs one run of skips instead of locking the guard (the 1-GPU smoke from
    initialisation, whose norms grew 100x within ten updates, skipped every update once the median was stale). The
    backstop fires only on a steady new scale: the last `max_consecutive` skipped norms and this one all lie within
    `factor` of each other. A burst is not a scale (EdgeGeo 80k retrain 2026-11-02: after 50 skips of norms from 1e8
    to 2e10 the backstop adopted them and applied a 5e10 update at step 28,846); it keeps being skipped, and the
    window slides until the norms settle. Once armed (`min_history` norms seen, or a scale adopted) the guard stays
    armed: an adopted history shorter than `min_history` must not switch detection off (round-1 audit 2026-11-02: with
    50/500/100 the 49 updates after an adoption went unchecked). DDP ranks hold the same averaged norm and so the
    same history; skip_flags still MAX-reduces the decision."""

    def __init__(self, factor=20.0, window=500, min_history=100, history=(), max_consecutive=500, skipped=(),
                 armed=False):
        self.factor, self.window, self.min_history = float(factor), int(window), int(min_history)
        self.max_consecutive, self.consecutive = int(max_consecutive), 0
        self.history = [float(value) for value in history][-self.window:]
        self.skipped = [float(value) for value in skipped][-self.window:]
        self.armed = bool(armed) or len(self.history) >= self.min_history
        self.adoptions = 0

    def median(self):
        if not self.history:
            return None
        ordered = sorted(self.history)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else .5 * (ordered[middle - 1] + ordered[middle])

    def is_spike(self, norm):
        """Decide this update (call once per update, before record)."""
        norm = float(norm)
        self.armed = self.armed or len(self.history) >= self.min_history
        spike = self.factor > 0 and math.isfinite(norm) and self.armed and bool(self.history) and \
            norm > self.factor * self.median()
        if spike and self.consecutive >= self.max_consecutive and self.steady(norm):  # backstop: adopt the new scale
            self.history, self.skipped, spike = self.skipped[-min(self.max_consecutive, self.window):], [], False
            self.adoptions += 1
        if not spike:
            self.skipped = []
        self.consecutive = self.consecutive + 1 if spike else 0
        return spike

    def steady(self, norm):
        """True when the last `max_consecutive` skipped norms (at most `window`) and `norm` span at most `factor`."""
        count = min(self.max_consecutive, self.window)
        recent = self.skipped[-count:] + [norm]
        return len(recent) > count and min(recent) > 0 and max(recent) <= self.factor * min(recent)

    def record(self, norm, applied=None):
        """Call after is_spike for the same update: a spike's norm goes to `skipped`, an applied one to `history`.
        `applied` is the final global decision (all skip reasons, all ranks); a norm of an update skipped for another
        reason (non-finite loss or cost) goes to neither. None: applied unless this guard called it a spike."""
        norm = float(norm)
        if not math.isfinite(norm):
            return
        if self.consecutive:
            target = self.skipped
        elif applied is None or applied:
            target = self.history
        else:
            return
        target.append(norm)
        del target[:-self.window]

    def state(self):
        return {'factor': self.factor, 'window': self.window, 'min_history': self.min_history,
                'max_consecutive': self.max_consecutive, 'consecutive': self.consecutive,
                'history': list(self.history), 'skipped': list(self.skipped), 'armed': self.armed,
                'adoptions': self.adoptions}


