"""Train the triage model, or score one set of vitals."""

from __future__ import annotations

import argparse
import math
import sys

from triage import __version__

EXIT_OK = 0
EXIT_ERROR = 1

EPILOG = """\
example:
  python -m triage predict --age 25 --systolic 130 --diastolic 80 \\
      --bs 15 --temp 98 --heart-rate 86

exit codes: 0 ok, 1 missing model or data, 2 bad arguments

A portfolio demo trained on 452 public records. Not a diagnostic device.
"""


def _positive(text: str) -> float:
    """argparse type: a finite number above zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description="Classify maternal health risk (low, mid, high) from six vitals.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub.add_parser(
        "train",
        help="clean the CSV, compare two forests, save the winner and its scores",
    )
    predict = sub.add_parser("predict", help="score one set of vitals")
    vitals = [
        ("--age", "years"),
        ("--systolic", "systolic blood pressure, mmHg"),
        ("--diastolic", "diastolic blood pressure, mmHg"),
        ("--bs", "blood sugar, mmol/L"),
        ("--temp", "body temperature, °F"),
        ("--heart-rate", "resting heart rate, bpm"),
    ]
    for flag, unit in vitals:
        predict.add_argument(
            flag, type=_positive, required=True, metavar="N", help=unit
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from triage.train import run

        return run()

    from triage.predict import ModelNotFoundError, predict_row

    try:
        result = predict_row(
            {
                "Age": args.age,
                "SystolicBP": args.systolic,
                "DiastolicBP": args.diastolic,
                "BS": args.bs,
                "BodyTemp": args.temp,
                "HeartRate": args.heart_rate,
            }
        )
    except ModelNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    ordered = sorted(
        result.probabilities.items(), key=lambda item: item[1], reverse=True
    )
    print(result.label)
    print("  " + "  ".join(f"{name} {score:.2f}" for name, score in ordered))
    return EXIT_OK
