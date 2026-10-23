#!/usr/bin/env python3
"""Give a cheap training-health verdict for Lightning CSVLogger output."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


EXIT_CODES = {"healthy": 0, "target_reached": 0, "plateau": 2,
              "diverged": 3, "no-data": 4}


def newest_metrics(run_dir):
    files = list(Path(run_dir).rglob("metrics.csv"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            return None
        eff_names = sorted(name for name in reader.fieldnames
                           if name.startswith("val/eff_"))
        steps = []
        loss = []
        efficiencies = {name: [] for name in eff_names}
        for row in reader:
            raw_step = row.get("step", "").strip()
            if not raw_step:
                continue
            try:
                step = int(float(raw_step))
            except ValueError:
                continue
            steps.append(step)
            raw_loss = row.get("val/loss", "").strip()
            if raw_loss:
                try:
                    loss.append((step, float(raw_loss)))
                except ValueError:
                    pass
            for name in eff_names:
                raw_value = row.get(name, "").strip()
                if raw_value:
                    try:
                        efficiencies[name].append((step, float(raw_value)))
                    except ValueError:
                        pass
    if not steps:
        return None
    return {"latest_step": max(steps), "loss": loss,
            "efficiencies": efficiencies}


def improvement(points, latest_step, window_steps):
    finite = [(step, value) for step, value in points if math.isfinite(value)]
    if not finite:
        return None, False
    latest = finite[-1]
    cutoff = latest_step - window_steps
    full_window = finite[0][0] <= cutoff
    candidates = [point for point in finite if point[0] <= cutoff]
    if not candidates:
        return None, full_window
    baseline = candidates[-1]
    return latest[1] - baseline[1], full_window


def throughput_estimate(files, newest_data):
    newest_mtime = files[0].stat().st_mtime
    newest_step = newest_data["latest_step"]
    for old_path in files[1:]:
        old_data = read_metrics(old_path)
        if not old_data:
            continue
        seconds = newest_mtime - old_path.stat().st_mtime
        step_delta = newest_step - old_data["latest_step"]
        if seconds > 0 and step_delta > 0:
            return step_delta / seconds
    return None


def evaluate(args):
    files = newest_metrics(args.run_dir)
    if not files:
        return {"verdict": "no-data", "error": "no metrics.csv found"}
    data = read_metrics(files[0])
    if not data:
        return {"verdict": "no-data", "metrics_csv": str(files[0]),
                "error": "metrics.csv has no usable step data"}

    result = {"metrics_csv": str(files[0]),
              "latest_step": data["latest_step"]}
    rate = throughput_estimate(files, data)
    if rate is not None:
        result["steps_per_sec"] = rate

    recent_loss = data["loss"][-5:]
    loss_values = [value for _, value in recent_loss]
    loss_summary = {"points": [{"step": step, "value": value}
                               for step, value in recent_loss], "slope": None}
    finite_loss = [(step, value) for step, value in recent_loss
                   if math.isfinite(value)]
    if len(finite_loss) >= 2:
        x_values, y_values = zip(*finite_loss)
        loss_summary["slope"] = float(np.polyfit(x_values, y_values, 1)[0])
    result["val_loss_trend"] = loss_summary

    eff_summary = {}
    full_windows = {}
    improvements = {}
    for name, points in data["efficiencies"].items():
        if not points:
            continue
        delta, full_window = improvement(points, data["latest_step"],
                                         args.window_steps)
        eff_summary[name] = {"latest": points[-1][1], "improvement": delta}
        improvements[name] = delta
        full_windows[name] = full_window
    result["efficiencies"] = eff_summary

    last_three = data["loss"][-3:]
    diverged = (any(not math.isfinite(value) for _, value in last_three) or
                (len(last_three) == 3 and
                 all(last_three[index][1] > last_three[index - 1][1]
                     for index in (1, 2))))
    plateau_names = ("val/eff_r30_75mm", "val/eff_r75plus")
    plateau_values = [improvements.get(name) for name in plateau_names]
    plateau = (all(value is not None for value in plateau_values) and
               any(full_windows.get(name, False) for name in plateau_names) and
               sum(plateau_values) < 0.005)

    if diverged:
        verdict = "diverged"
    elif plateau:
        verdict = "plateau"
    elif args.target_steps is not None and data["latest_step"] >= args.target_steps:
        verdict = "target_reached"
    else:
        verdict = "healthy"
    result["verdict"] = verdict
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--window-steps", type=int, default=30000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.window_steps <= 0:
        parser.error("--window-steps must be positive")

    result = evaluate(args)
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=True))
    else:
        for key, value in result.items():
            if key != "verdict":
                print(f"{key}: {value}")
        print(f"VERDICT: {result['verdict']}")
    return EXIT_CODES[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
