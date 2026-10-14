"""Enter circuit design values and predict outputs with a saved OTA model."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_ota import MLP, MultiTaskMLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "2stage": PROJECT_DIR / "data" / "2stage" / "2stage.csv",
    "2stage_full": PROJECT_DIR / "data" / "2stage" / "2stage.csv",
    "5tr": PROJECT_DIR / "data" / "5tr" / "5tr.csv",
    "5tr_full": PROJECT_DIR / "data" / "5tr" / "ota_dataset_3000.csv",
}
# CSV/model values use SI units; terminal values use the units shown here.
INPUT_DISPLAY = {
    **{name: ("um", 1e-6) for name in
       ("L01", "L23", "L56", "W01", "W23", "W56", "W4", "L4", "W7", "L7",
        "Wm", "Wn", "Wp", "Li", "Lii", "Lm")},
    "Ibias": ("uA", 1e-6), "I_bias": ("uA", 1e-6), "Ib": ("uA", 1e-6),
    "R_c": ("kOhm", 1e3), "R_C": ("kOhm", 1e3),
    "C_c": ("pF", 1e-12), "Cc": ("pF", 1e-12),
}


def display_unit(name: str) -> tuple[str, float]:
    """Unknown inputs remain in their CSV SI units rather than using a guessed conversion."""
    return INPUT_DISPLAY.get(name, ("SI", 1.0))


def in_range(value: float, bounds: tuple[float, float]) -> bool:
    """Allow tiny float32 rounding at a recorded training boundary."""
    low, high = bounds
    tolerance = 1e-6 * max(abs(low), abs(high), 1.0)
    return low - tolerance <= value <= high + tolerance
TARGET_DISPLAY = {
    "DCGAIN_dB": ("dB", 1.0, 3),
    "UGF_Hz": ("MHz", 1e-6, 3),
    "PM_deg": ("deg", 1.0, 3),
    "IDD_A": ("uA", 1e6, 3),
    "Vout_V": ("V", 1.0, 4),
    **{f"M{i}_sat_V": ("V", 1.0, 4) for i in range(8)},
}
DISPLAY_OUTPUTS = {
    "2stage": ["DCGAIN_dB", "UGF_Hz", "PM_deg", "IDD_A", "Vout_V",
               *(f"M{i}_sat_V" for i in range(8))],
    "5tr": ["DCGAIN_dB", "UGF_Hz", "PM_deg", "IDD_A", "Vout_V",
            *(f"M{i}_sat_V" for i in range(6))],
    "2stage_full": ["DCGAIN_dB", "UGF_Hz", "PM_deg", "IDD_A", "Vout_V",
                    *(f"M{i}_sat_V" for i in range(8)),
                    *(f"M{i}_region" for i in range(8))],
    "5tr_full": ["DCGAIN_dB", "UGF_Hz", "PM_deg", "IDD_A", "Vout_V",
                 *(f"M{i}_sat_V" for i in range(6)),
                 *(f"M{i}_region" for i in range(6))],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=DATA_FILES, required=True, help="Choose the saved circuit model")
    parser.add_argument("--model-dir", type=Path,
                        help="Use an updated model folder instead of runs/<preset>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--values", nargs="+", type=float, metavar="NUMBER",
                      help="Inputs in the displayed order and units")
    mode.add_argument("--random", action="store_true",
                      help="Pick a random point within each training input range")
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible --random values")
    return parser.parse_args()


def load_saved_model(preset: str, model_dir: Path | None = None):
    folder = model_dir or PROJECT_DIR / "runs" / preset
    try:
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        x_scaler = joblib.load(folder / "x_scaler.joblib")
        y_scaler = joblib.load(folder / "y_scaler.joblib")
    except FileNotFoundError as exc:
        raise SystemExit(f"Saved {preset} model is missing: {exc.filename}") from exc
    inputs, targets = checkpoint["inputs"], checkpoint["targets"]
    if checkpoint.get("model_type") == "multitask":
        model = MultiTaskMLP(checkpoint["n_inputs"], checkpoint["n_targets"],
                             len(checkpoint["binary_targets"]),
                             [len(classes) for classes in checkpoint["region_classes"]],
                             checkpoint["hidden"])
    else:
        model = MLP(checkpoint["n_inputs"], checkpoint["n_targets"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, x_scaler, y_scaler, inputs, targets, checkpoint


def get_training_ranges(preset: str, inputs: list[str]) -> dict[str, tuple[float, float]]:
    # The CSV stores SI units; convert each input to its displayed unit.
    data = pd.read_csv(DATA_FILES[preset], usecols=inputs)
    if data.empty:
        raise SystemExit(f"No rows found in {DATA_FILES[preset]}")
    return {name: (float(data[name].min()) / display_unit(name)[1],
                   float(data[name].max()) / display_unit(name)[1])
            for name in inputs}


def choose_values(args: argparse.Namespace, inputs: list[str], ranges: dict[str, tuple[float, float]]) -> list[float]:
    print(f"\n{args.preset} training input ranges:")
    for name in inputs:
        low, high = ranges[name]
        print(f"  {name:6s} {low:g}-{high:g} {display_unit(name)[0]}")

    if args.random:
        rng = random.Random(args.seed)
        return [rng.uniform(*ranges[name]) for name in inputs]
    if args.values is not None:
        if len(args.values) != len(inputs):
            raise SystemExit(f"Expected {len(inputs)} values in this order: {', '.join(inputs)}")
        return args.values

    values = []
    print("\nEnter one value for each input using the unit shown in the prompt.")
    for name in inputs:
        low, high = ranges[name]
        while True:
            try:
                value = float(input(f"{name} ({display_unit(name)[0]}, {low:g}-{high:g}): ").strip())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                values.append(value)
                break
            except ValueError:
                print("Enter a positive number, for example 0.8 or 12.6.")
    return values


def main() -> None:
    # Keep Korean status text readable in Windows and VS Code terminals.
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    model, x_scaler, y_scaler, inputs, targets, checkpoint = load_saved_model(args.preset, args.model_dir)
    saved_ranges = checkpoint.get("input_ranges_si")
    ranges = ({name: (bounds[0] / display_unit(name)[1],
                      bounds[1] / display_unit(name)[1])
               for name, bounds in saved_ranges.items()} if saved_ranges
              else get_training_ranges(args.preset, inputs))
    values = choose_values(args, inputs, ranges)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise SystemExit("All inputs must be positive finite numbers.")

    print("\nInput values:")
    for name, value in zip(inputs, values):
        print(f"  {name:6s} {value:g} {display_unit(name)[0]}")
        low, high = ranges[name]
        if not in_range(value, (low, high)):
            print(f"  WARNING: {name} is outside the training range {low:g}-{high:g} {display_unit(name)[0]}.")

    # Convert each displayed unit (um, uA, kOhm, pF, ...) to the CSV's SI unit.
    x = np.asarray([[value * display_unit(name)[1]
                     for name, value in zip(inputs, values)]], dtype=np.float32)
    scaled_x = x_scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        result = model(torch.from_numpy(scaled_x))
    if checkpoint.get("model_type") == "multitask":
        scaled_y, binary_logits, region_logits = result
        binary_prob = torch.sigmoid(binary_logits).numpy()[0]
        binary_names = checkpoint["binary_targets"]
        region_names = checkpoint["region_targets"]
        flags = {name: (int(prob >= 0.5), float(prob))
                 for name, prob in zip(binary_names, binary_prob)}
        region_codes = {name: checkpoint["region_classes"][col][int(logits.argmax(dim=1)[0])]
                        for col, (name, logits) in enumerate(zip(region_names, region_logits))}
        scaled_y = scaled_y.numpy()
    else:
        scaled_y = result.numpy()
        flags, region_codes = {}, {}
    predicted = y_scaler.inverse_transform(scaled_y)[0]

    print("\nPredicted outputs:")
    available = dict(zip(targets, predicted))
    for name in dict.fromkeys([*DISPLAY_OUTPUTS[args.preset], *targets,
                               *flags, *region_codes]):
        if name in flags:
            flag, probability = flags[name]
            print(f"  {name:12s} {flag} (0/1 flag, P(1)={probability:.3f})")
            continue
        if name in region_codes:
            print(f"  {name:12s} {region_codes[name]} (region code)")
            continue
        if name not in available:
            print(f"  {name:12s} 값 없음 (이 모델에서 학습하지 않음)")
            continue
        unit, multiplier, digits = TARGET_DISPLAY.get(name, ("SI units", 1.0, 6))
        print(f"  {name:12s} {available[name] * multiplier:.{digits}f} {unit}")
    if any(not in_range(value, ranges[name]) for name, value in zip(inputs, values)):
        print("\nOutside-range predictions are extrapolations; check them with Virtuoso before using them.")


if __name__ == "__main__":
    main()
