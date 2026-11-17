"""
AP9SI - Step 1: preprocessing of the ReV-StED recordings.

What it does:
  * keeps only onboard (CAN, *_obd) inputs + the Correvit sideslip target
  * resamples 50 Hz -> 10 Hz by averaging 0.1 s time bins (handles time gaps)
  * removes standstill: mean wheel speed < 7.2 km/h (= 2 m/s), using onboard
    wheel speeds only, so the filter could also run in a production car
  * Recording 3 = FINAL TEST SET: cleaned the same way but saved to its own
    file; it is NOT used for any statistics or plots here.

Outputs (relative to the car_dataset folder):
  processed/dev_10hz.csv          recordings 1,2,4,5,6,7 (development set)
  processed/test_rec3_10hz.csv    recording 3 (final test, do not open until the end)
  results/recording_overview.csv
  results/summary_stats_dev.csv
  figures/fig1_sideslip_hist_dev.png
  figures/fig2_sideslip_vs_latacc_dev.png
  figures/fig3_sideslip_by_recording_dev.png

Run:  python 01_preprocess.py   (from the code folder)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"   # put the 7 ReV-StED recording CSVs here
OUT = BASE / "processed"
FIG = BASE / "figures"
RES = BASE / "results"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

TEST_RECORDING = 3                      
DEV_RECORDINGS = [1, 2, 4, 5, 6, 7]     
BIN_S = 0.1                             # 10 Hz
MIN_SPEED_KMH = 7.2                     # 2 m/s

INPUTS = ["Yawrate_obd", "LatAcc_obd", "SW_pos_obd", "VelFL_obd", "VelFR_obd",
          "VelRL_obd", "VelRR_obd", "speedo_obd", "brake_pressure_obd"]  
TARGET = "Correvit_vehiclesideslip_angle_COG"                            
TIME = "INS_time_sec"


def clean_recording(n: int):
    raw = pd.read_csv(RAW / f"Recording_{n}_ReVSTED.csv", usecols=[TIME] + INPUTS + [TARGET])
    t0 = raw[TIME].iloc[0]
    raw["bin"] = np.floor((raw[TIME] - t0) / BIN_S + 1e-6).astype(int)
    d = raw.groupby("bin").mean().reset_index(drop=True)          # 10 Hz, empty bins (gaps) simply absent
    d["t_s"] = d[TIME] - t0                                        # seconds since start of recording
    n10 = len(d)
    wheel = d[["VelFL_obd", "VelFR_obd", "VelRL_obd", "VelRR_obd"]].mean(axis=1)
    d = d[wheel >= MIN_SPEED_KMH].copy()
    d.insert(0, "recording", n)
    d = d[["recording", "t_s"] + INPUTS + [TARGET]]
    info = dict(recording=n,
                role="final test" if n == TEST_RECORDING else "development",
                raw_rows_50hz=len(raw),
                raw_minutes=round((raw[TIME].iloc[-1] - t0) / 60, 1),
                rows_10hz=n10,
                rows_kept_moving=len(d),
                kept_minutes=round(len(d) * BIN_S / 60, 1),
                removed_standstill_pct=round(100 * (1 - len(d) / n10), 1))
    return d, info


dev_parts, overview = [], []
for n in sorted(DEV_RECORDINGS + [TEST_RECORDING]):
    d, info = clean_recording(n)
    overview.append(info)
    if n == TEST_RECORDING:
        d.to_csv(OUT / "test_rec3_10hz.csv", index=False) # locked away
    else:
        dev_parts.append(d)
    print(f"Recording {n}: {info['rows_kept_moving']} rows kept ({info['kept_minutes']} min)")

dev = pd.concat(dev_parts, ignore_index=True)
dev.to_csv(OUT / "dev_10hz.csv", index=False)
pd.DataFrame(overview).to_csv(RES / "recording_overview.csv", index=False)

# descriptive statistics: DEVELOPMENT SET ONLY 
stats = dev[INPUTS + [TARGET]].describe(percentiles=[.25, .5, .75]).T
stats = stats.rename(columns={"50%": "median", "25%": "q25", "75%": "q75"})
stats["missing"] = dev[INPUTS + [TARGET]].isna().sum()
stats.round(3).to_csv(RES / "summary_stats_dev.csv")

# figures: DEVELOPMENT SET ONLY 
BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
y = dev[TARGET]

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(y, bins=np.arange(-15, 15.25, 0.25), color=BLUE, edgecolor=SURF, linewidth=0.3)
for v in (-2, 2):
    ax.axvline(v, color=INK2, linewidth=1, linestyle="--")
share = (y.abs() > 2).mean() * 100
ax.text(2.4, ax.get_ylim()[1] * 0.85, f"|sideslip| > 2°: {share:.0f}% of samples", color=INK2, fontsize=9)
ax.set_xlabel("Vehicle sideslip angle (°)")
ax.set_ylabel("Samples (10 Hz)")
ax.set_title("Sideslip angle is mostly small (development set)", loc="left", fontsize=11, color=INK)
fig.savefig(FIG / "fig1_sideslip_hist_dev.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
hb = ax.hexbin(dev["LatAcc_obd"], y, gridsize=70, bins="log", cmap="Blues", mincnt=1)
cb = fig.colorbar(hb, ax=ax); cb.set_label("Samples (log scale)", color=INK2)
r = np.corrcoef(dev["LatAcc_obd"], y)[0, 1]
ax.set_xlabel("Onboard lateral acceleration (m/s²)")
ax.set_ylabel("Vehicle sideslip angle (°)")
ax.set_title(f"Sideslip vs lateral acceleration (development set, r = {r:.2f})", loc="left", fontsize=11, color=INK)
fig.savefig(FIG / "fig2_sideslip_vs_latacc_dev.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
groups = [dev.loc[dev.recording == n, TARGET] for n in DEV_RECORDINGS]
ax.boxplot(groups, tick_labels=[f"Rec {n}" for n in DEV_RECORDINGS], showfliers=False, widths=0.5,
           patch_artist=True, boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE),
           medianprops=dict(color=BLUE, linewidth=2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
ax.set_ylabel("Vehicle sideslip angle (°)")
ax.set_title("Sideslip distribution per development recording (outliers hidden)", loc="left", fontsize=11, color=INK)
fig.savefig(FIG / "fig3_sideslip_by_recording_dev.png"); plt.close(fig)

print(f"\nDevelopment set: {len(dev)} rows, {len(dev) * BIN_S / 3600:.2f} h")
print("Saved to", OUT, "and", FIG)
