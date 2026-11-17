"""
AP9SI - Step 2: model comparison with leave-one-recording-out cross-validation.

Uses ONLY the development set (processed/dev_10hz.csv, recordings 1,2,4,5,6,7).
The final test set (Recording 3) is not read by this script.

What it does:
  * 4 models: linear regression, random forest, gradient boosting, MLP
  * a small grid of settings per model
  * 6 folds: train on 5 development recordings, validate on the 6th
  * metrics per fold: RMSE, MAE (all samples) and RMSE on |sideslip| > 2 deg
  * best settings per model = lowest mean RMSE across the 6 folds
  * Friedman test on the 6 fold RMSEs of the best settings,
    then Nemenyi post-hoc test (critical difference of mean ranks, Demsar 2006)

Outputs (relative to the car_dataset folder):
  results/cv_all_configs.csv      every setting x fold
  results/cv_best_per_fold.csv    best setting per model, per fold
  results/cv_summary.csv          mean/std per model (best settings)
  results/best_settings.json      chosen settings (used later for the final test)
  results/stats_tests.txt         Friedman + Nemenyi output
  figures/fig4_cv_rmse_per_fold.png
  figures/fig5_critical_difference.png

Requires: pandas, numpy, scikit-learn, scipy, matplotlib
Run:  python 02_train_cv.py   (from the code folder)
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "processed" / "dev_10hz.csv"
RES = BASE / "results"
FIG = BASE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = 42
INPUTS = ["Yawrate_obd", "LatAcc_obd", "SW_pos_obd", "VelFL_obd", "VelFR_obd",
          "VelRL_obd", "VelRR_obd", "speedo_obd", "brake_pressure_obd"]
TARGET = "Correvit_vehiclesideslip_angle_COG"
HIGH_SLIP = 2.0   # degrees, threshold for the "high sideslip" error

# models and their small tuning grids
def grid():
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


dev = pd.read_csv(DATA)
recs = sorted(dev.recording.unique())
assert 3 not in recs, "Recording 3 is the final test set and must not be here"
print(f"Development set: {len(dev)} rows, recordings {recs}")

rows = []
for model_name, setting, model in grid():
    for fold, val_rec in enumerate(recs, start=1):
        tr, va = dev[dev.recording != val_rec], dev[dev.recording == val_rec]
        t0 = time.time()
        model.fit(tr[INPUTS], tr[TARGET])
        pred = model.predict(va[INPUTS])
        y = va[TARGET].to_numpy()
        hi = np.abs(y) > HIGH_SLIP
        rows.append(dict(model=model_name, setting=setting, fold=fold, val_recording=val_rec,
                         n_train=len(tr), n_val=len(va),
                         rmse=rmse(y, pred), mae=float(np.mean(np.abs(y - pred))),
                         rmse_high_slip=rmse(y[hi], pred[hi]), fit_seconds=round(time.time() - t0, 1)))
        print(f"{model_name:18s} | {setting:38s} | fold {fold} (rec {val_rec}) "
              f"RMSE {rows[-1]['rmse']:.3f}  MAE {rows[-1]['mae']:.3f}  [{rows[-1]['fit_seconds']}s]", flush=True)

allcv = pd.DataFrame(rows)
allcv.to_csv(RES / "cv_all_configs.csv", index=False)

#best setting per model = lowest mean RMSE over the 6 folds 
mean_rmse = allcv.groupby(["model", "setting"]).rmse.mean().reset_index()
best = mean_rmse.loc[mean_rmse.groupby("model").rmse.idxmin()]
best_settings = dict(zip(best.model, best.setting))
json.dump(best_settings, open(RES / "best_settings.json", "w"), indent=2)
bestcv = allcv.merge(best[["model", "setting"]], on=["model", "setting"])
bestcv.to_csv(RES / "cv_best_per_fold.csv", index=False)

order = ["Linear regression", "Random forest", "Gradient boosting", "MLP"]
summary = (bestcv.groupby("model")[["rmse", "mae", "rmse_high_slip", "fit_seconds"]]
           .agg(["mean", "std"]).round(3).reindex(order))
summary.to_csv(RES / "cv_summary.csv")

# Friedman + Nemenyi on fold RMSEs 
wide = bestcv.pivot(index="val_recording", columns="model", values="rmse")[order]
fr = stats.friedmanchisquare(*[wide[m] for m in order])
ranks = wide.rank(axis=1)                       # rank 1 = lowest RMSE within a fold
mean_ranks = ranks.mean().sort_values()
k, n = len(order), len(wide)
q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
pairs = []
for i in range(k):
    for j in range(i + 1, k):
        a, b = order[i], order[j]
        diff = abs(mean_ranks[a] - mean_ranks[b])
        pairs.append((a, b, diff, diff > cd))

with open(RES / "stats_tests.txt", "w") as f:
    f.write("Model comparison on 6 leave-one-recording-out folds (development set)\n")
    f.write("Metric: RMSE (deg) of the best setting per model\n\n")
    f.write(wide.round(4).to_string() + "\n\n")
    f.write("H0: all models have the same median RMSE (equal mean ranks)\n")
    f.write(f"Friedman chi-square = {fr.statistic:.3f}, df = {k - 1}, p = {fr.pvalue:.4g}\n")
    f.write(f"Decision at alpha = 0.05: {'reject H0' if fr.pvalue < 0.05 else 'do not reject H0'}\n\n")
    f.write("Mean ranks (1 = best):\n" + mean_ranks.round(3).to_string() + "\n\n")
    f.write(f"Nemenyi post-hoc: k = {k}, N = {n}, q_0.05 = {q_alpha:.3f}, critical difference CD = {cd:.3f}\n")
    for a, b, d, sig in pairs:
        f.write(f"  {a} vs {b}: |rank diff| = {d:.3f} -> {'significant' if sig else 'not significant'}\n")
print(open(RES / "stats_tests.txt").read())
print(summary)

# figures 
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLORS = {"Linear regression": "#2a78d6", "Random forest": "#eb6834",
          "Gradient boosting": "#1baf7a", "MLP": "#eda100"}
MARKERS = {"Linear regression": "o", "Random forest": "s", "Gradient boosting": "D", "MLP": "^"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(n)
for m in order:
    ax.plot(x, wide[m], marker=MARKERS[m], color=COLORS[m], linewidth=2, markersize=7,
            markeredgecolor=SURF, markeredgewidth=1.5, label=m)
ax.set_xticks(x, [f"Rec {r}" for r in wide.index])
ax.set_xlabel("Validation recording (fold)")
ax.set_ylabel("RMSE (°)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
ax.set_title("Validation RMSE per fold, best setting per model", loc="left", fontsize=11, color=INK)
fig.savefig(FIG / "fig4_cv_rmse_per_fold.png"); plt.close(fig)

# critical difference diagram (Demsar 2006)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.set_xlim(0.7, k + 0.3); ax.set_ylim(-1.9, 0.9); ax.axis("off")
ax.hlines(0, 1, k, color=INK2, linewidth=1)
for r in range(1, k + 1):
    ax.vlines(r, -0.06, 0.06, color=INK2, linewidth=1)
    ax.text(r, 0.12, str(r), ha="center", va="bottom", color=INK2, fontsize=9)
ax.hlines(0.65, 1, 1 + cd, color=INK, linewidth=2)
ax.vlines([1, 1 + cd], 0.6, 0.7, color=INK, linewidth=2)
ax.text(1 + cd + 0.08, 0.65, f"CD = {cd:.2f}  (mean rank, lower = better)", va="center", color=INK2, fontsize=9)
# models whose mean ranks lie within CD of each other are joined by a bar (no significant difference)
ranked = list(mean_ranks.items())
cliques = []
for i in range(k):
    j = max(jj for jj in range(i, k) if ranked[jj][1] - ranked[i][1] < cd)
    if j > i and not any(a <= i and j <= b for a, b in cliques):
        cliques.append((i, j))
for c, (i, j) in enumerate(cliques):
    yb = -0.22 - 0.16 * c
    ax.hlines(yb, ranked[i][1] - 0.04, ranked[j][1] + 0.04, color=INK, linewidth=3)
# labels: best models to the left, worst to the right, lines never cross
left = [(m, r) for m, r in ranked if r <= (1 + k) / 2]
right = [(m, r) for m, r in ranked if r > (1 + k) / 2][::-1]
for side, items in (("left", left), ("right", right)):
    for i, (m, r) in enumerate(items):
        y = -0.85 - 0.3 * i
        ax.plot([r, r], [0, y], color=COLORS[m], linewidth=1.5)
        ax.plot(r, 0, marker=MARKERS[m], color=COLORS[m], markersize=7, markeredgecolor=SURF, zorder=3)
        end = 0.75 if side == "left" else k + 0.25
        ax.hlines(y, r, end, color=COLORS[m], linewidth=1.5)
        ax.text(end - 0.03 if side == "left" else end + 0.03, y, f"{m} ({r:.2f})",
                ha="right" if side == "left" else "left", va="center", color=INK, fontsize=9)
ax.set_title("Critical difference diagram (Nemenyi, α = 0.05)",
             loc="left", fontsize=11, color=INK, pad=12)
fig.savefig(FIG / "fig5_critical_difference.png"); plt.close(fig)
print("Saved results to", RES, "and figures to", FIG)
