# Virtuoso CSV로 OTA 성능 예측하기

2-stage amplifier와 5TR OTA는 **서로 다른 CSV와 모델**로 학습합니다. 두 회로가 공유하는 것은 `train_ota.py`의 학습 절차뿐입니다.

GitHub에는 코드, Virtuoso CSV, 학습 모델과 결과(`runs/`)를 함께 올립니다. 로컬 Python 환경만 포함하지 않습니다. 새로 받은 사람은 Python 패키지를 설치한 뒤 제공된 데이터로 변환, 학습, 예측을 다시 실행할 수 있습니다.

## 데이터 구조

```text
ota_ml_project/
├─ train_ota.py                    # 공통 학습 코드
├─ predict_ota.py                  # 새 설계 조건의 성능 예측
├─ prepare_2stage.py              # Virtuoso 원본을 학습용 표로 변환
├─ prepare_5tr.py                 # 5TR 원본을 학습용 표로 변환
├─ requirements.txt
├─ data/
│  ├─ 2stage/
│  │  ├─ raw/2stage_raw.csv        # Virtuoso 원본 CSV
│  │  └─ 2stage.csv                # 변환된 학습용 CSV
│  └─ 5tr/
│     ├─ raw/5tr_raw.csv          # Virtuoso 원본 CSV
│     └─ 5tr.csv                  # 변환된 학습용 CSV
└─ runs/
   ├─ 2stage/                     # 2-stage 모델과 결과
   └─ 5tr/                        # 5TR 모델과 결과
```

처음 받은 2-stage 원본 파일 이름에는 `3600`이 들어 있었지만 실제 조합은 **3,584개**입니다. 두 회로 모두 `raw/*_raw.csv`가 Virtuoso에서 나온 가로로 긴 원본이고, 회로 이름의 `.csv`가 학습용 표입니다. `2stage.csv`는 한 행이 하나의 시뮬레이션 조건이며, 열은 다음과 같습니다.

| 역할 | CSV 열 | 단위 |
| --- | --- | --- |
| 입력 | `L01`, `L23`, `L56` | m |
| 입력 | `Ibias` | A |
| 예측값 | `DCGAIN_dB` | dB |
| 예측값 | `UGF_Hz` | Hz |
| 예측값 | `PM_deg` | degree |
| 예측값 | `IDD_A` | A |

원본에서 다시 표를 만들려면 프로젝트 폴더의 VS Code 터미널에서 실행합니다.

```powershell
.\.venv_ready\Scripts\python.exe prepare_2stage.py
```

## 2-stage 학습 실행

VS Code에서 **Terminal → New Terminal**을 열고, 현재 폴더가 `ota_ml_project`인지 확인한 뒤 실행합니다. 프로젝트 안에 준비된 `.venv_ready` Python을 사용합니다. VS Code에서 Python 선택이 필요하면 **Python: Select Interpreter**에서 `.venv_ready\Scripts\python.exe`를 선택하세요.

```powershell
.\.venv_ready\Scripts\python.exe train_ota.py --preset 2stage --csv data\2stage\2stage.csv --epochs 300 --patience 30
```

CSV 경로는 `--csv` 뒤에서 지정합니다. Python 코드 안에 CSV 내용을 붙여 넣을 필요는 없습니다. 현재 2-stage 열 이름은 기본 설정에 반영되어 있어 `--inputs`와 `--targets`를 생략할 수 있습니다. 다른 CSV를 쓰면 실제 헤더와 같은 이름을 지정하세요.

```powershell
.\.venv_ready\Scripts\python.exe train_ota.py --preset 2stage --csv data\2stage\other.csv --inputs L01 L23 L56 Ibias --targets DCGAIN_dB UGF_Hz PM_deg IDD_A
```

`--inputs`는 설계 변수, `--targets`는 예측할 성능입니다. 대소문자, 밑줄까지 CSV 첫 줄의 열 이름과 같아야 합니다. 열 이름에 공백이 있으면 따옴표로 묶습니다.

## 결과 확인

`runs/2stage/`와 `runs/5tr/`에 각각 다음 파일이 저장됩니다.

- `model.pt`: 학습한 PyTorch 모델과 모델 설정
- `x_scaler.joblib`, `y_scaler.joblib`: 입력과 출력의 StandardScaler
- `metrics.json`: train, validation, test의 출력별 MAE, RMSE, R²
- `predictions.csv`: 각 행의 실제값, 예측값, 데이터 분할

저장된 2-stage 실행 결과는 학습 2,508행, 검증 538행, 테스트 538행입니다. 무작위 분할은 시뮬레이션 격자 안의 보간 성능을 보여줍니다. 격자 밖의 설계나 공정 조건까지 정확하다는 뜻은 아닙니다.

저장된 5TR 실행 결과는 학습 2,195행, 검증 470행, 테스트 471행입니다. 테스트 MAE는 PM 0.0178°, IDD 0.110 µA입니다. 나머지 6개 출력의 지표는 `runs/5tr/metrics.json`에서 확인할 수 있습니다.

## 5TR OTA 학습 실행

넣어 주신 5TR 원본은 가로로 넓은 Virtuoso 형식입니다. `prepare_5tr.py`가 이를 **3,136행**으로 바꾸어 `data/5tr/5tr.csv`에 저장했습니다. 입력은 `Li`, `Lii`, `Lm`(길이, m), `Ib`(전류, A)입니다. 출력은 `PM_deg`, `IDD_A`, `M0_sat_V`부터 `M5_sat_V`까지 총 8개입니다. 원본에는 Gain과 UGF가 없습니다.

원본에서 다시 변환할 때:

```powershell
.\.venv_ready\Scripts\python.exe prepare_5tr.py
```

5TR만 따로 학습할 때:

```powershell
.\.venv_ready\Scripts\python.exe train_ota.py --preset 5tr --csv data\5tr\5tr.csv --epochs 300 --patience 30
```

5TR 결과는 `runs/5tr/`에 저장되며 2-stage 모델을 덮어쓰지 않습니다. `--targets`로 일부 출력만 선택할 수도 있습니다. 다른 CSV를 사용할 때는 그 파일의 열 이름에 맞게 `--inputs`와 `--targets`를 지정하세요.

## 새 입력값으로 예측하기

`predict_ota.py`는 저장된 모델과 두 StandardScaler를 불러옵니다. 길이는 **µm**, 전류는 **µA**로 입력합니다. 예를 들어 `0.3`은 길이에서 0.3 µm이며, `8`은 전류에서 8 µA입니다. 값을 입력하지 않으면 터미널에서 하나씩 묻습니다.

```powershell
.\.venv_ready\Scripts\python.exe predict_ota.py --preset 2stage
.\.venv_ready\Scripts\python.exe predict_ota.py --preset 5tr
```

값을 한 번에 지정하거나 범위 안에서 무작위로 고를 수도 있습니다. `--values`의 순서는 화면에 표시되는 입력 순서와 같습니다.

```powershell
.\.venv_ready\Scripts\python.exe predict_ota.py --preset 2stage --values 0.3 0.3 0.35 8
.\.venv_ready\Scripts\python.exe predict_ota.py --preset 5tr --values 1.8 1.35 3.3 10.33
.\.venv_ready\Scripts\python.exe predict_ota.py --preset 2stage --random
```

| 회로 | 입력 | 실제 학습 범위 |
| --- | --- | --- |
| 2-stage | `L01`, `L23` | 각각 0.3–3.8 µm |
| 2-stage | `L56` | 0.35–3.85 µm |
| 2-stage | `Ibias` | 8–20 µA |
| 5TR | `Li`, `Lm` | 각각 0.3–3.8 µm |
| 5TR | `Lii` | 0.35–3.35 µm |
| 5TR | `Ib` | 3–25 µA |

현재 데이터는 이 범위 안의 일정한 sweep 점들로 구성됩니다. 범위 안의 임의 값은 주로 **보간**, 범위를 벗어난 값은 **외삽**입니다. 스크립트는 범위 밖 입력에 경고를 표시하지만 예측 자체는 출력합니다. 범위 밖의 결과는 Virtuoso로 확인하고, 그 영역에서도 정확한 모델이 필요하다면 시뮬레이션 데이터를 더 모아 다시 학습해야 합니다.

## 새 PC에서 설치하기

이 PC에서는 `.venv_ready`가 이미 준비되어 있습니다. 다른 PC로 프로젝트를 복사할 때는 새 Python 환경을 만든 뒤 패키지를 설치합니다.

```powershell
py -m venv .venv_new
.\.venv_new\Scripts\python.exe -m pip install -r requirements.txt
```

그 PC에서는 실행 명령의 `.venv_ready`를 `.venv_new`로 바꾸세요. `.venv_ready`는 이 PC의 Python 설치를 참조하므로 다른 PC로 복사해 그대로 사용할 수 없습니다.
