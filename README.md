# ML virtual sensor for vehicle sideslip angle

Code and results for the paper **"Comparing Machine Learning Models as Virtual Sensors for Vehicle Sideslip Angle Estimation from Onboard Signals"** (Bc. Adéla Polková, course AP9SI Experimental Methods in Software Engineering, Faculty of Applied Informatics, Tomas Bata University in Zlín, 2026).

The vehicle sideslip angle is important for stability control, but measuring it needs optical sensors that are too expensive for production cars. This project compares four machine learning models — **linear regression, random forest, gradient boosting and a multilayer perceptron (MLP)** — that estimate the sideslip angle from nine standard onboard (CAN-bus) signals of the public [ReV-StED dataset](https://zenodo.org/records/15270060).

## Results at a glance

| Model | Cross-validation RMSE (°), mean ± std | Final test RMSE (°), Recording 3 | Final test R² |
| --- | --- | --- | --- |
| MLP | 0.253 ± 0.042 | 0.234 | 0.9936 |
| Random forest | 0.303 ± 0.126 | **0.225** | 0.9940 |
| Gradient boosting | 0.292 ± 0.088 | 0.241 | 0.9931 |
| Linear regression | 0.363 ± 0.116 | 0.307 | 0.9889 |

- Friedman test on the 6 cross-validation folds: χ² = 15.0, p = 0.0018 → the models do not perform equally.
- Nemenyi post-hoc test (CD = 1.915): only **MLP vs linear regression** differs significantly.
- On the held-out recording, random forest, MLP and gradient boosting are practically tied; all clearly beat linear regression.

## Method in short

1. **Inputs:** 9 onboard signals only — yaw rate, lateral acceleration, steering wheel angle, 4 wheel speeds, speedometer, brake pressure. Reference-sensor (ADMA/INS) columns are never used as inputs.
2. **Target:** sideslip angle at the centre of gravity measured by the Correvit sensor.
3. **Preprocessing:** 50 Hz → 10 Hz (0.1 s bin averages); standstill removed (mean onboard wheel speed < 7.2 km/h).
4. **Split:** Recording 3 is the final test set, used once at the end. Recordings 1, 2, 4, 5, 6, 7 are used for 6-fold leave-one-recording-out cross-validation (tuning + model comparison).
5. **Statistics:** Friedman test and Nemenyi post-hoc test on fold RMSEs (Demšar, 2006).

## Repository structure

```
code/
  01_preprocess.py   clean the 7 recordings, split dev/test, summary stats, Fig. 1-3
  02_train_cv.py     tune + compare models with leave-one-recording-out CV, Friedman/Nemenyi, Fig. 4-5
  03_final_test.py   retrain on all dev recordings, evaluate once on Recording 3, Fig. 6-7
results/             all metrics, chosen settings, test statistics, logs and test-set predictions
figures/             the 7 figures used in the paper
requirements.txt
```

## How to reproduce

1. Install the requirements: `pip install -r requirements.txt`
2. Download the 7 recording files (`Recording_1_ReVSTED.csv` … `Recording_7_ReVSTED.csv`) from [Zenodo](https://zenodo.org/records/15270060) and put them in `data/raw/`.
3. Run the scripts in order from the `code` folder:
   ```
   cd code
   python 01_preprocess.py
   python 02_train_cv.py
   python 03_final_test.py
   ```
   Outputs are written to `processed/` (cleaned data, not tracked by Git), `results/` and `figures/`. All random processes use seed 42. Training takes a few minutes on a laptop.

## Data used

This project uses the **ReV-StED — Real-world Vehicle State Estimation Dataset**, created and published by the authors of the paper below. All credit for collecting and publishing the data goes to them. The data is **not** included in this repository; download it from the links below.

**Researchers (authors of the dataset paper):**
Abinav Kalyanasundaram, Karthikeyan Chandra Sekaran, Philipp Stäuber, Michael Lange, Wolfgang Utschick and Michael Botsch.

**About the data:** about 5 hours of driving with a Smart Fortwo test vehicle on the CARISSMA outdoor test track in Ingolstadt, Germany, driven by five test drivers. It contains time-synchronised measurements from a GeneSys ADMA GNSS/inertial system, a Correvit optical sensor and the vehicle's own onboard (CAN-bus) sensors.

| Source | Link | What it contains | Used here |
| --- | --- | --- | --- |
| Full dataset (Zenodo) | https://zenodo.org/records/15270060 | The 7 full recordings `Recording_1_ReVSTED.csv` … `Recording_7_ReVSTED.csv` | Yes — all 7 recordings (put them in `data/raw/`) |
| Official GitHub repository of the dataset (MB-Team-THI) | https://github.com/MB-Team-THI/UAHL-RevStED | Sample files `ADMA_data_sample.csv` and `OBD_Sample.csv`, the data dictionary `ReV-StED Description.xlsx` (every column with its unit), example videos | Data dictionary used to select the input and target columns |
| Dataset paper (arXiv) | https://arxiv.org/abs/2504.06105 | Description of the dataset, sensors and synchronisation, and the authors' own hybrid machine learning method (UAHL) | Background, benchmark results for comparison |

**Dataset paper:**
A. Kalyanasundaram, K. Chandra Sekaran, P. Stäuber, M. Lange, W. Utschick and M. Botsch, *Uncertainty-Aware Hybrid Machine Learning in Virtual Sensors for Vehicle Sideslip Angle Estimation*, arXiv:2504.06105, 2025 (submitted to the IEEE Intelligent Vehicles Symposium 2025).

**If you use the data, please cite the dataset and the paper above.** The exact dataset citation (with DOI) is shown on the Zenodo record page.

### How this project uses the data

- Inputs: only the 9 onboard signals (`Yawrate_obd`, `LatAcc_obd`, `SW_pos_obd`, `VelFL_obd`, `VelFR_obd`, `VelRL_obd`, `VelRR_obd`, `speedo_obd`, `brake_pressure_obd`).
- Target: `Correvit_vehiclesideslip_angle_COG` (sideslip angle at the centre of gravity, from the Correvit sensor).
- The reference-sensor (ADMA/INS) columns are not used as inputs.
- Recording 3 is the final test set; recordings 1, 2, 4, 5, 6 and 7 are used for training and cross-validation.

## Licence

- **Code** in this repository: MIT License (see `LICENSE`).
- **ReV-StED data**: Creative Commons Attribution 4.0 International (CC BY 4.0), owned by its authors. The files in `results/` and `figures/` are derived from ReV-StED and are shared under the same attribution requirement.

## Acknowledgements

Many thanks to Abinav Kalyanasundaram, Karthikeyan Chandra Sekaran, Philipp Stäuber, Michael Lange, Wolfgang Utschick and Michael Botsch for creating ReV-StED and making it publicly available.
