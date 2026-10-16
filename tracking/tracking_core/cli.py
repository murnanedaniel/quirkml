"""Shared evaluator arguments; importing this module needs only the standard library."""
import argparse
from collections.abc import Sequence

from .summary import SELECTIONS

DEFAULT_MIN_MEASUREMENTS = 7


def add_eval_args(
    parser: argparse.ArgumentParser, *, num_events: int | None = None, num_threads: int = 4,
    min_measurements: int = DEFAULT_MIN_MEASUREMENTS, selections: Sequence[str] = SELECTIONS,
    working_points: bool = True,
) -> None:
    """Add common evaluation options with the eval-checkout defaults."""
    group = parser.add_argument_group("ACTS evaluation")
    group.add_argument("--num-events", "--events", type=int, default=num_events,
                       help="Number of events to evaluate after --skip-events.")
    group.add_argument("--num-threads", "--threads", type=int, default=num_threads,
                       help="ACTS Sequencer threads; affects wall time, not event selection.")
    group.add_argument("--skip-events", type=int, default=0,
                       help="Skip this many input events, retaining original event positions for remapping.")
    group.add_argument("--pt-min-gev", type=float, default=1.0, help="Minimum truth-particle pT in GeV.")
    group.add_argument("--eta-max", type=float, default=3.0, help="Maximum absolute truth-particle eta.")
    group.add_argument("--bfield-tesla", type=float, default=3.0, help="Solenoid field in tesla.")
    group.add_argument("--min-measurements", type=int, default=min_measurements,
                       help="Minimum measurements for technical-selection particles and reconstructed tracks.")
    group.add_argument("--selections", nargs="+", choices=SELECTIONS, default=list(selections),
                       help="Particle selections; technical is always retained for fit diagnostics.")
    group.add_argument("--working-points", action=argparse.BooleanOptionalAction, default=working_points,
                       help="Also evaluate loose, tight and perfect matching; otherwise use very loose only.")


def eval_kwargs(args: argparse.Namespace) -> dict:
    """Translate shared options to dump_eval_histograms keywords; fixtures may resolve num_events afterwards."""
    from .eval import WORKING_POINTS

    return {
        "num_events": args.num_events, "num_threads": args.num_threads, "skip_events": args.skip_events,
        "pt_min_gev": args.pt_min_gev, "eta_max": args.eta_max, "bfield_tesla": args.bfield_tesla,
        "min_measurements": args.min_measurements, "selections": args.selections,
        "working_points": WORKING_POINTS if args.working_points else None,
    }
