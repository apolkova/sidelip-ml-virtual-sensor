"""
AP9SI - Step 3: FINAL TEST on Recording 3 (run once).

  * retrains each model on ALL development recordings (1,2,4,5,6,7)
    with the settings chosen in cross-validation (results/best_settings.json)
  * predicts the final test set (processed/test_rec3_10hz.csv) once
  * nothing is tuned or changed after this step

Outputs (relative to the car_dataset folder):
  results/final_test_metrics.csv       RMSE, MAE, max error, R2, error at high/low sideslip
  results/final_test_predictions.csv   time, true sideslip, prediction of each model
  figures/fig6_final_pred_vs_true.png  predicted vs measured sideslip, one panel per model
  figures/fig7_final_timeseries.png    30 s excerpt of Recording 3, measured vs predicted

Requires: pandas, numpy, scikit-learn, matplotlib
Run:  python 03_final_test.py   (from the code folder)
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RES, FIG = BASE / "results", BASE / "figures"
SEED = 42
INPUTS = ["Yawrate_obd", "LatAcc_obd", "SW_pos_obd", "VelFL_obd", "VelFR_obd",
          "VelRL_obd", "VelRR_obd", "speedo_obd", "brake_pressure_obd"]
TARGET = "Correvit_vehiclesideslip_angle_COG"
HIGH_SLIP = 2.0
ORDER = ["Linear regression", "Random forest", "Gradient boosting", "MLP"]


def candidates():
    """Same candidate settings as 02_train_cv.py."""
    yield "Linear regression", "default", make_pipeline(StandardScaler(), LinearRegression())
    for depth, leaf in [(10, 5), (20, 5), (None, 20)]:
        yield ("Random forest", f"max_depth={depth}, min_samples_leaf={leaf}",
               RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                                     max_samples=0.3, n_jobs=-1, random_state=SEED))
    for lr, leaves in [(0.05, 31), (0.1, 31), (0.1, 63)]:
        yield ("Gradient boosting", f"learning_rate={lr}, max_leaf_nodes={leaves}",
               HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=500,
                                             early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=20, random_state=SEED))
    for hidden, alpha in [((64, 64), 1e-4), ((128, 64), 1e-4), ((128, 64), 1e-3)]:
        yield ("MLP", f"hidden={hidden}, alpha={alpha}",
               make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, batch_size=256,
                                          learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=8,
                                          random_state=SEED)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


best = json.load(open(RES / "best_settings.json"))
dev = pd.read_csv(BASE / "processed" / "dev_10hz.csv")
test = pd.read_csv(BASE / "processed" / "test_rec3_10hz.csv")
assert set(dev.recording) == {1, 2, 4, 5, 6, 7} and set(test.recording) == {3}
y = test[TARGET].to_numpy()
hi, lo = np.abs(y) > HIGH_SLIP, np.abs(y) <= HIGH_SLIP
print(f"Train: {len(dev)} rows (recordings 1,2,4,5,6,7) | Test: {len(test)} rows (recording 3)")

preds, rows = {}, []
for name, setting, model in candidates():
    if best[name] != setting:
        continue
    t0 = time.time()
    model.fit(dev[INPUTS], dev[TARGET])
    fit_s = time.time() - t0
    p = model.predict(test[INPUTS])
    preds[name] = p
    rows.append(dict(model=name, setting=setting,
                     rmse=rmse(y, p), mae=float(np.mean(np.abs(y - p))),
                     max_abs_error=float(np.max(np.abs(y - p))),
                     r2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                     rmse_high_slip=rmse(y[hi], p[hi]), rmse_low_slip=rmse(y[lo], p[lo]),
                     fit_seconds=round(fit_s, 1)))
    print(f"{name:18s} RMSE {rows[-1]['rmse']:.3f}  MAE {rows[-1]['mae']:.3f}  "
          f"RMSE|>2deg| {rows[-1]['rmse_high_slip']:.3f}  R2 {rows[-1]['r2']:.4f}")

metrics = pd.DataFrame(rows).set_index("model").reindex(ORDER)
metrics.to_csv(RES / "final_test_metrics.csv", float_format="%.4f")
out = pd.DataFrame({"t_s": test.t_s, "true_sideslip": y, **{f"pred_{m}": preds[m] for m in ORDER}})
out.to_csv(RES / "final_test_predictions.csv", index=False, float_format="%.4f")
print(f"\nHigh-sideslip samples (|beta| > {HIGH_SLIP} deg): {hi.sum()} of {len(y)} ({hi.mean():.1%})")
print(metrics.round(4).to_string())

# figures 
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLORS = {"Linear regression": "#2a78d6", "Random forest": "#eb6834",
          "Gradient boosting": "#1baf7a", "MLP": "#eda100"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

lim = np.ceil(max(np.abs(y).max(), max(np.abs(p).max() for p in preds.values())))
fig, axes = plt.subplots(2, 2, figsize=(7, 6.6), sharex=True, sharey=True)
for ax, m in zip(axes.flat, ORDER):
    ax.scatter(y, preds[m], s=3, alpha=0.25, color=COLORS[m], linewidths=0, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, linewidth=1, linestyle="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{m}  (RMSE {metrics.loc[m, 'rmse']:.3f}°)", loc="left", fontsize=10, color=INK)
for ax in axes[1]:
    ax.set_xlabel("Measured sideslip (°)")
for ax in axes[:, 0]:
    ax.set_ylabel("Predicted sideslip (°)")
fig.suptitle("Final test on Recording 3: predicted vs measured sideslip", x=0.02, ha="left",
             fontsize=11, color=INK)
fig.savefig(FIG / "fig6_final_pred_vs_true.png"); plt.close(fig)

# 30 s excerpt: the window with the largest sideslip variation (chosen for illustration only)
win = 300
roll = pd.Series(y).rolling(win).std()
end = int(roll.idxmax()) + 1
seg = slice(end - win, end)
t = test.t_s.to_numpy()[seg] - test.t_s.to_numpy()[seg][0]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(t, y[seg], color=INK, linewidth=2.2, label="Measured (Correvit)")
for m in ORDER:
    ax.plot(t, preds[m][seg], color=COLORS[m], linewidth=1.4, label=m)
ax.set_xlabel("Time in excerpt (s)")
ax.set_ylabel("Sideslip (°)")
ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.2))
ax.set_title("Recording 3, 30 s excerpt with the strongest sideslip changes", loc="left",
             fontsize=11, color=INK)
fig.savefig(FIG / "fig7_final_timeseries.png"); plt.close(fig)
print("Saved results to", RES, "and figures to", FIG)
