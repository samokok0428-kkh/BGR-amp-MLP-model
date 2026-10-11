"""Convert a wide Virtuoso 2-stage sweep export into one row per simulation."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


HEADER = re.compile(
    r'^value\(value\(value\((DCGAIN|UGF|PM|IDD) "L01" ([0-9.eE+-]+)\) '
    r'"L23" ([0-9.eE+-]+)\) "L56" ([0-9.eE+-]+)\)(?: \([^)]*\))?$'
)
METRIC_COLUMNS = {
    "DCGAIN": "DCGAIN_dB",
    "UGF": "UGF_Hz",
    "PM": "PM_deg",
    "IDD": "IDD_A",
}
# These outputs were not exported by Virtuoso. Keep empty columns so the schema
# shows that they are unavailable, without inventing measurement values.
UNMEASURED_COLUMNS = ["Vout_V", *(f"M{i}_sat_V" for i in range(8))]
OUTPUT_COLUMNS = ["L01", "L23", "L56", "Ibias", *METRIC_COLUMNS.values(), *UNMEASURED_COLUMNS]


def convert(source: Path, destination: Path) -> tuple[int, Counter]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        if headers[0].strip() != "I":
            raise ValueError(f"Expected first header 'I', got {headers[0]!r}")

        # Each Virtuoso column encodes the metric and the three swept lengths.
        # Reorganize those columns by one (L01, L23, L56) combination.
        column_map: dict[tuple[float, float, float], dict[str, int]] = {}
        for col_index, header in enumerate(headers[1:], start=1):
            match = HEADER.fullmatch(header.strip())
            if match is None:
                raise ValueError(f"Unrecognized Virtuoso header at column {col_index + 1}: {header}")
            metric = match.group(1)
            lengths = tuple(float(part) for part in match.groups()[1:])
            group = column_map.setdefault(lengths, {})
            if metric in group:
                raise ValueError(f"Duplicate {metric} for {lengths}")
            group[metric] = col_index

        expected = set(METRIC_COLUMNS)
        for lengths, group in column_map.items():
            if set(group) != expected:
                raise ValueError(f"Incomplete outputs for {lengths}: {sorted(group)}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        with destination.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(OUTPUT_COLUMNS)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise ValueError(f"Row {row_number}: expected {len(headers)} cells, got {len(row)}")
                bias = float(row[0])
                if not math.isfinite(bias):
                    raise ValueError(f"Row {row_number}: invalid bias current {row[0]!r}")
                for lengths, group in column_map.items():
                    values = []
                    for metric in METRIC_COLUMNS:
                        value = float(row[group[metric]])
                        if not math.isfinite(value):
                            raise ValueError(f"Row {row_number}: non-finite {metric} at {lengths}")
                        values.append(value)
                    writer.writerow([*lengths, bias, *values, *([""] * len(UNMEASURED_COLUMNS))])
                    counts["simulations"] += 1
                counts["bias_values"] += 1
        counts["length_combinations"] = len(column_map)
    return counts["simulations"], counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/2stage/raw/2stage_raw.csv"), help="Original wide Virtuoso CSV")
    parser.add_argument("--out", type=Path, default=Path("data/2stage/2stage.csv"), help="Prepared row-wise CSV")
    args = parser.parse_args()
    n, counts = convert(args.raw, args.out)
    print(f"Wrote {n} rows to {args.out}")
    print(dict(counts))


if __name__ == "__main__":
    main()
