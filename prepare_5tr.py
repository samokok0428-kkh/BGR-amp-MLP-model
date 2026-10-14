
"""Convert the wide Virtuoso 5TR export into one row per simulation."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


# The original headers contain the measured output plus Li, Lii, and Lm.
HEADER = re.compile(
    r'^value\(value\(value\(([A-Za-z0-9_]+) "Li" ([0-9.eE+-]+)\) '
    r'"Lii" ([0-9.eE+-]+)\) "Lm" ([0-9.eE+-]+)\)(?: \([^)]*\))?$'
)
METRIC_COLUMNS = {
    "PM": "PM_deg",
    "IDD": "IDD_A",
    "M0_sat": "M0_sat_V",
    "M1_sat": "M1_sat_V",
    "M2_sat": "M2_sat_V",
    "M3_sat": "M3_sat_V",
    "M4_sat": "M4_sat_V",
    "M5_sat": "M5_sat_V",
}
# Gain, UGF, and Vout are absent from this Virtuoso export.
UNMEASURED_COLUMNS = ["DCGAIN_dB", "UGF_Hz", "Vout_V"]
OUTPUT_COLUMNS = ["Li", "Lii", "Lm", "Ib", *METRIC_COLUMNS.values(), *UNMEASURED_COLUMNS]


def convert(source: Path, destination: Path) -> Counter:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        if headers[0].strip() != "Ib":
            raise ValueError(f"Expected first header 'Ib', got {headers[0]!r}")

        columns: dict[tuple[float, float, float], dict[str, int]] = {}
        for col_index, header in enumerate(headers[1:], start=1):
            match = HEADER.fullmatch(header.strip())
            if match is None:
                raise ValueError(f"Unrecognized header at column {col_index + 1}: {header}")
            metric = match.group(1)
            if metric not in METRIC_COLUMNS:
                raise ValueError(f"Unexpected metric at column {col_index + 1}: {metric}")
            lengths = tuple(float(part) for part in match.groups()[1:])
            group = columns.setdefault(lengths, {})
            if metric in group:
                raise ValueError(f"Duplicate {metric} for {lengths}")
            group[metric] = col_index

        expected = set(METRIC_COLUMNS)
        for lengths, group in columns.items():
            if set(group) != expected:
                raise ValueError(f"Incomplete outputs for {lengths}: {sorted(group)}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        seen_bias = set()
        with destination.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(OUTPUT_COLUMNS)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise ValueError(f"Row {row_number}: expected {len(headers)} cells, got {len(row)}")
                bias = float(row[0])
                if not math.isfinite(bias) or bias in seen_bias:
                    raise ValueError(f"Row {row_number}: invalid or repeated bias {row[0]!r}")
                seen_bias.add(bias)
                for lengths, group in columns.items():
                    values = []
                    for metric in METRIC_COLUMNS:
                        value = float(row[group[metric]])
                        if not math.isfinite(value):
                            raise ValueError(f"Row {row_number}: non-finite {metric} at {lengths}")
                        values.append(value)
                    writer.writerow([*lengths, bias, *values, *([""] * len(UNMEASURED_COLUMNS))])
                    counts["simulations"] += 1
                counts["bias_values"] += 1
        counts["length_combinations"] = len(columns)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/5tr/raw/5tr_raw.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/5tr/5tr.csv"))
    args = parser.parse_args()
    counts = convert(args.raw, args.out)
    print(f"Wrote {counts['simulations']} rows to {args.out}")
    print(dict(counts))


if __name__ == "__main__":
    main()
