"""Print a live progress bar for a training run, read from MLflow.

Strictly read-only: it polls the tracking server and prints, and never writes a
metric, a tag or an artifact. Training itself is unchanged, so the numbers a run
logs are the numbers it would have logged without this script.

Progress is read off the metrics the trainers already emit:

  * SASRec streams ``train_step_loss`` every 450 steps, so the step count gives
    a fraction of the estimated total (``n_train_rows / examples_per_step`` per
    epoch).
  * The two-tower emits nothing per step, only ``epoch_warm_recall_at_k_candidates``
    once per epoch, so its bar advances in whole epochs.

Usage::

    python scripts/training_progress.py                 # newest run in the experiment
    python scripts/training_progress.py --run-name sasrec
    python scripts/training_progress.py --run-id <id>
    python scripts/training_progress.py --watch          # refresh until it finishes
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

EXPERIMENT = "phase-2-candidates"
TRACKING_URI = "http://localhost:5000"
BAR_WIDTH = 28

STEP_METRIC = "train_step_loss"
EPOCH_METRIC = "epoch_warm_recall_at_k_candidates"
# Examples are drawn from interactions and short sequences yield none, so the
# step count sits a little under ``rows / batch``. The estimate only positions
# the bar; the epoch metric is what actually says an epoch is finished.
EXAMPLES_PER_ROW = 0.96


def resolve_run(client: MlflowClient, run_id: str | None, name: str | None) -> Run:
    if run_id:
        return client.get_run(run_id)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"no MLflow experiment named {EXPERIMENT!r}")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    for run in runs:
        if name and name not in run.data.tags.get("mlflow.runName", ""):
            continue
        return run
    raise SystemExit(f"no run in {EXPERIMENT!r} matches {name!r}")


def progress(run: Run, client: MlflowClient) -> tuple[float, int, int, int]:
    """Return (fraction, epochs completed, current step, estimated total steps)."""
    if run.info.status == "FINISHED":
        epochs = int(float(run.data.params.get("epochs", 1)) or 1)
        steps = client.get_metric_history(run.info.run_id, STEP_METRIC)
        return 1.0, epochs, max((row.step for row in steps), default=0), 0

    params = run.data.params
    epochs = int(float(params.get("epochs", 1)) or 1)
    completed = len(client.get_metric_history(run.info.run_id, EPOCH_METRIC))
    current_step = max(
        (row.step for row in client.get_metric_history(run.info.run_id, STEP_METRIC)),
        default=0,
    )
    if current_step and completed:
        per_epoch = max(1, current_step // completed)
    elif current_step:
        rows = float(params.get("n_train_rows", 0) or 0)
        batch = float(params.get("examples_per_step", 512) or 512)
        per_epoch = max(1, int(rows * EXAMPLES_PER_ROW / batch))
    else:
        return min(1.0, completed / epochs), completed, current_step, 0
    total_steps = per_epoch * epochs
    return min(1.0, current_step / total_steps), completed, current_step, total_steps


def draw(fraction: float, run: Run, client: MlflowClient) -> None:
    run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
    params = run.data.params
    metrics = run.data.metrics
    epochs = int(float(params.get("epochs", 1)) or 1)
    started = datetime.fromtimestamp(run.info.start_time / 1000)
    elapsed = (run.info.end_time or int(time.time() * 1000)) / 1000 - run.info.start_time / 1000
    _fraction, completed, current_step, total_steps = progress(run, client)

    filled = int(BAR_WIDTH * min(1.0, fraction))
    bar = "#" * filled + "." * (BAR_WIDTH - filled)

    status = run.info.status
    eta = ""
    if fraction > 0 and status == "RUNNING":
        eta = f"eta {_duration(elapsed * (1 - fraction) / fraction)}"
    elif status != "RUNNING":
        eta = f"total {_duration(elapsed)}"

    loss = metrics.get("train_loss")
    recall = metrics.get(EPOCH_METRIC)

    print(f"{run_name}  [{status}]")
    print(f"[{bar}] {fraction * 100:5.1f}%  epoch {min(completed, epochs)}/{epochs}"
          + (f"  step {current_step}/{total_steps}" if total_steps else "")
          + (f"  loss {loss:.4f}" if loss is not None else "")
          + (f"  warm recall@500 {recall:.4f}" if recall is not None else ""))
    print(f"started {started:%m-%d %H:%M}  elapsed {_duration(elapsed)}  {eta}".rstrip())


def _duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, _ = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--watch", action="store_true", help="refresh until the run finishes")
    parser.add_argument("--interval", type=int, default=30, help="seconds between refreshes")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    args = parser.parse_args(argv)

    try:
        client = MlflowClient(args.tracking_uri)
        while True:
            run = resolve_run(client, args.run_id, args.run_name)
            fraction, _completed, _step, _total = progress(run, client)
            draw(fraction, run, client)
            if not args.watch or run.info.status != "RUNNING":
                return 0
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - this is a status tool, not a gate
        print(f"cannot read MLflow at {args.tracking_uri}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
