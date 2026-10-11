"""Enter circuit design values and predict outputs with a saved OTA model."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_ota import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "2stage": PROJECT_DIR / "data" / "2stage" / "2stage.csv",
    "5tr": PROJECT_DIR / "data" / "5tr" / "5tr.csv",
}
# Every current input is a length or current. Both display units have a 1e-6 SI multiplier.
INPUT_UNITS = {
    "L01": "um", "L23": "um", "L56": "um", "Ibias": "uA",
    "Li": "um", "Lii": "um", "Lm": "um", "Ib": "uA",
}
TARGET_DISPLAY = {
    "DCGAIN_dB": ("dB", 1.0, 3),
    "UGF_Hz": ("MHz", 1e-6, 3),
    "PM_deg": ("deg", 1.0, 3),
    "IDD_A": ("uA", 1e6, 3),
    **{f"M{i}_sat_V": ("V", 1.0, 4) for i in range(6)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=DATA_FILES, required=True, help="Choose the saved circuit model")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--values", nargs="+", type=float, metavar="NUMBER",
                      help="Inputs in the displayed order, using um for lengths and uA for current")
    mode.add_argument("--random", action="store_true",
                      help="Pick a random point within each training input range")
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible --random values")
    return parser.parse_args()


def load_saved_model(preset: str):
    folder = PROJECT_DIR / "runs" / preset
    try:
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        x_scaler = joblib.load(folder / "x_scaler.joblib")
        y_scaler = joblib.load(folder / "y_scaler.joblib")
    except FileNotFoundError as exc:
        raise SystemExit(f"Saved {preset} model is missing: {exc.filename}") from exc
    inputs, targets = checkpoint["inputs"], checkpoint["targets"]
    model = MLP(checkpoint["n_inputs"], checkpoint["n_targets"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, x_scaler, y_scaler, inputs, targets


def get_training_ranges(preset: str, inputs: list[str]) -> dict[str, tuple[float, float]]:
    # The prepared CSV stores SI units (m and A). Convert to um and uA for input.
    data = pd.read_csv(DATA_FILES[preset], usecols=inputs)
    if data.empty:
        raise SystemExit(f"No rows found in {DATA_FILES[preset]}")
    return {name: (float(data[name].min()) * 1e6, float(data[name].max()) * 1e6)
            for name in inputs}


def choose_values(args: argparse.Namespace, inputs: list[str], ranges: dict[str, tuple[float, float]]) -> list[float]:
    print(f"\n{args.preset} training input ranges:")
    for name in inputs:
        low, high = ranges[name]
        print(f"  {name:6s} {low:g}-{high:g} {INPUT_UNITS[name]}")

    if args.random:
        rng = random.Random(args.seed)
        return [rng.uniform(*ranges[name]) for name in inputs]
    if args.values is not None:
        if len(args.values) != len(inputs):
            raise SystemExit(f"Expected {len(inputs)} values in this order: {', '.join(inputs)}")
        return args.values

    values = []
    print("\nEnter one value for each input. Use um for lengths and uA for current.")
    for name in inputs:
        low, high = ranges[name]
        while True:
            try:
                value = float(input(f"{name} ({INPUT_UNITS[name]}, {low:g}-{high:g}): ").strip())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                values.append(value)
                break
            except ValueError:
                print("Enter a positive number, for example 0.8 or 12.6.")
    return values


def main() -> None:
    args = parse_args()
    model, x_scaler, y_scaler, inputs, targets = load_saved_model(args.preset)
    ranges = get_training_ranges(args.preset, inputs)
    values = choose_values(args, inputs, ranges)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise SystemExit("All inputs must be positive finite numbers.")

    print("\nInput values:")
    for name, value in zip(inputs, values):
        print(f"  {name:6s} {value:g} {INPUT_UNITS[name]}")
        low, high = ranges[name]
        if not low <= value <= high:
            print(f"  WARNING: {name} is outside the training range {low:g}-{high:g} {INPUT_UNITS[name]}.")

    # Convert displayed um/uA values to the SI units used during training.
    x = np.asarray([values], dtype=np.float32) * 1e-6
    scaled_x = x_scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        scaled_y = model(torch.from_numpy(scaled_x)).numpy()
    predicted = y_scaler.inverse_transform(scaled_y)[0]

    print("\nPredicted outputs:")
    for name, value in zip(targets, predicted):
        unit, multiplier, digits = TARGET_DISPLAY.get(name, ("SI units", 1.0, 6))
        print(f"  {name:12s} {value * multiplier:.{digits}f} {unit}")
    if any(not ranges[name][0] <= value <= ranges[name][1] for name, value in zip(inputs, values)):
        print("\nOutside-range predictions are extrapolations; check them with Virtuoso before using them.")


if __name__ == "__main__":
    main()
