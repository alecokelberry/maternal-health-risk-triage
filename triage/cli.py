"""Train the triage model, or score one set of vitals."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triage",
        description="Rank maternal-health follow-up priority from routine vitals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Clean the CSV, compare forests, and save the winner")
    predict = sub.add_parser("predict", help="Score one row with the saved model")
    predict.add_argument("--age", type=float, required=True)
    predict.add_argument("--systolic", type=float, required=True)
    predict.add_argument("--diastolic", type=float, required=True)
    predict.add_argument("--bs", type=float, required=True, help="Blood sugar, mmol/L")
    predict.add_argument("--temp", type=float, required=True, help="Body temperature, Fahrenheit")
    predict.add_argument("--heart-rate", type=float, required=True)

    args = parser.parse_args(argv)
    if args.command == "train":
        from triage.train import run

        return run()

    from triage.predict import predict_row

    label, probabilities, warnings = predict_row(
        {
            "Age": args.age,
            "SystolicBP": args.systolic,
            "DiastolicBP": args.diastolic,
            "BS": args.bs,
            "BodyTemp": args.temp,
            "HeartRate": args.heart_rate,
        }
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    print(label)
    print("  " + "  ".join(f"{name} {probability:.2f}" for name, probability in ordered))
    return 0
