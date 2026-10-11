
"""Virtuoso sweep CSV 한 개로 여러 회로 성능을 동시에 예측하는 MLP를 학습한다."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 현재 CSV의 기본 컬럼명. inputs는 설계변수(X), targets는 예측할 회로 성능(Y)이다.
# 실제 CSV 헤더가 다르면 실행할 때 --inputs, --targets로 정확한 이름을 지정한다.
PRESETS = {
    "2stage": {
        "inputs": ["L01", "L23", "L56", "Ibias"],
        "targets": ["DCGAIN_dB", "UGF_Hz", "PM_deg", "IDD_A"],
    },
    "5tr": {
        "inputs": ["Li", "Lii", "Lm", "Ib"],
        "targets": ["PM_deg", "IDD_A", "M0_sat_V", "M1_sat_V", "M2_sat_V", "M3_sat_V", "M4_sat_V", "M5_sat_V"],
    },
}

# Virtuoso CSV 값이 "300n", "12.6uA", "21.4MHz"처럼 저장될 수 있다.
# 정규식은 숫자 / SI 접두어 / 단위 문자를 분리한다. 읽을 수 없는 값은 아래에서 NaN으로 처리한다.
NUMBER = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(meg|[fpnumkKMGTPµμ]?)\s*(?:[A-Za-z°Ω/]+)?\s*$"
)
# SI 접두어를 실제 숫자로 바꾸는 배수. m=10^-3, M=10^6처럼 대소문자가 중요하다.
SCALE = {
    "": 1.0, "f": 1e-15, "p": 1e-12, "n": 1e-9,
    "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3,
    "k": 1e3, "K": 1e3, "meg": 1e6, "M": 1e6,
    "G": 1e9, "T": 1e12, "P": 1e15,
}


def to_number(value: object) -> float:
    """예: 300n→3e-7, 12.6u→1.26e-5. 변환 실패 시 NaN을 돌려준다."""
    if pd.isna(value):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = NUMBER.fullmatch(str(value).strip())
    if not match:
        return math.nan
    return float(match.group(1)) * SCALE[match.group(2)]


class MLP(nn.Module):
    """입력층 → 은닉층 2개(ReLU) → 출력층으로 구성한 다중 출력 회귀 모델."""

    def __init__(self, n_inputs: int, n_targets: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 한 행의 설계변수에서 지정된 성능값들을 동시에 예측한다.
        return self.net(x)


def parse_args() -> argparse.Namespace:
    """터미널 실행 옵션을 읽는다. --csv만 필수이고 나머지는 기본값이 있다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Virtuoso export CSV")
    parser.add_argument("--out", type=Path, help="Result folder (default: runs/<preset>)")
    parser.add_argument("--preset", choices=PRESETS, default="2stage")
    parser.add_argument("--inputs", nargs="+", help="Exact CSV input column names")
    parser.add_argument("--targets", nargs="+", help="Exact CSV output column names")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """데이터 행 번호를 훈련 70%, 검증 15%, 테스트 15%로 나눈다."""
    if n_rows < 20:
        raise ValueError(f"Only {n_rows} usable rows; at least 20 are needed for a 70/15/15 split.")
    indices = np.arange(n_rows)
    # 같은 seed면 분할을 재현할 수 있다. test는 최종 성능 확인용이다.
    train, rest = train_test_split(indices, test_size=0.30, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed)
    return np.asarray(train), np.asarray(val), np.asarray(test)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> dict:
    """출력 변수마다 실제 단위 기준 MAE, RMSE, R²를 계산한다."""
    result = {}
    for col, name in enumerate(targets):
        actual, predicted = y_true[:, col], y_pred[:, col]
        result[name] = {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        }
    return result


def main() -> None:
    # 1) 실행 옵션과 CSV 컬럼 설정을 확인한다.
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.hidden < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, hidden, and batch-size must be positive.")
    inputs = args.inputs or PRESETS[args.preset]["inputs"]
    targets = args.targets or PRESETS[args.preset]["targets"]
    if not inputs or not targets:
        raise ValueError(
            f"{args.preset} needs both --inputs and --targets with the exact CSV headers."
        )
    out = args.out or Path("runs") / args.preset
    columns = inputs + targets
    if len(columns) != len(set(columns)):
        raise ValueError("Input and target column names must be unique.")

    # 2) 난수 시드를 고정하고 CSV를 읽는다. 헤더 앞뒤 공백은 제거한다.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = pd.read_csv(args.csv, encoding=args.encoding)
    raw.columns = raw.columns.str.strip()
    missing = [col for col in columns if col not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}\nAvailable columns: {raw.columns.tolist()}\n"
            "Pass the exact headers with --inputs and --targets."
        )

    # 3) 단위 접두어를 숫자로 바꾸고, 입력/출력 중 하나라도 실패한 행은 제외한다.
    # 실패 행 수는 metrics.json에 기록한다. 이 회귀 모델은 실패 여부를 학습하지 않는다.
    numeric = raw[columns].apply(lambda series: series.map(to_number))
    valid = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=numeric.index)
    clean = numeric.loc[valid].copy()
    dropped = int((~valid).sum())
    # 4) 데이터 분할 후 입력 X와 정답 Y를 float32 배열로 만든다.
    train_idx, val_idx, test_idx = split_indices(len(clean), args.seed)
    x = clean[inputs].to_numpy(dtype=np.float32)
    y = clean[targets].to_numpy(dtype=np.float32)
    # 5) StandardScaler는 평균을 빼고 표준편차로 나눈다.
    # fit은 훈련 행에만 적용해야 검증/테스트 데이터 정보가 학습에 새지 않는다.
    # 출력도 스케일링하면 단위가 크게 다른 성능값들이 손실 함수에 고르게 반영된다.
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit(x[train_idx])
    y_scaler.fit(y[train_idx])
    xs = x_scaler.transform(x).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)

    # 6) 미니배치로 훈련 데이터를 섞어 공급한다. 검증 데이터는 학습에 사용하지 않는다.
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xs[train_idx]), torch.from_numpy(ys[train_idx])),
        batch_size=args.batch_size, shuffle=True,
    )
    # Adam으로 가중치를 갱신하고, 표준화된 출력의 평균제곱오차(MSE)를 최소화한다.
    model = MLP(len(inputs), len(targets), args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    xv, yv = torch.from_numpy(xs[val_idx]), torch.from_numpy(ys[val_idx])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    # 7) 매 epoch 뒤 검증 손실을 확인한다. patience 동안 개선이 없으면 조기 종료한다.
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # 평가에서는 역전파용 계산을 만들지 않는다.
            val_loss = float(loss_fn(model(xv), yv))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    # 마지막 epoch가 아니라 검증 성능이 가장 좋았던 가중치를 사용한다.
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # 예측값을 원래 단위(dB, Hz, A, V 등)로 되돌린다.
        pred = y_scaler.inverse_transform(model(torch.from_numpy(xs)).numpy())

    # 8) 원본 CSV 행 번호와 train/validation/test 구분을 예측 CSV에 함께 기록한다.
    splits = np.empty(len(clean), dtype=object)
    splits[train_idx], splits[val_idx], splits[test_idx] = "train", "validation", "test"
    predictions = clean.copy()
    predictions.insert(0, "source_csv_row", clean.index.to_numpy() + 2)
    predictions.insert(1, "split", splits)
    for col, name in enumerate(targets):
        predictions[f"pred_{name}"] = pred[:, col]
    # 각 분할의 오차를 별도로 기록한다. 모델 비교에는 보지 않은 test 지표를 사용한다.
    report = {
        "csv": str(args.csv), "inputs": inputs, "targets": targets,
        "source_rows": len(raw), "used_rows": len(clean), "dropped_invalid_rows": dropped,
        "split_rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "seed": args.seed, "best_epoch": best_epoch,
        "metrics": {
            "train": metrics(y[train_idx], pred[train_idx], targets),
            "validation": metrics(y[val_idx], pred[val_idx], targets),
            "test": metrics(y[test_idx], pred[test_idx], targets),
        },
    }
    # 9) 모델, 두 스케일러, 행별 예측값, 평가 지표를 출력 폴더에 저장한다.
    # 나중에 새 데이터를 예측할 때도 같은 스케일러와 모델을 함께 사용해야 한다.
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_inputs": len(inputs),
                "n_targets": len(targets), "hidden": args.hidden,
                "inputs": inputs, "targets": targets}, out / "model.pt")
    joblib.dump(x_scaler, out / "x_scaler.joblib")
    joblib.dump(y_scaler, out / "y_scaler.joblib")
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved results to {out.resolve()}")
    print(f"Rows: {len(clean)} used, {dropped} invalid/dropped. Best epoch: {best_epoch}")
    print(json.dumps(report["metrics"]["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 이 파일을 `python train_ota.py ...`로 직접 실행했을 때만 main()을 시작한다.
    main()
