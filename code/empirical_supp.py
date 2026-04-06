import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


# Path settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(DATA_DIR, "figure", "supplement")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# Input files and event windows
CASES = {
    "Subprime_KOSPI": {
        "file": "subprime_KOSPI.csv",
        "title": "KOSPI during the Global Financial Crisis",
        "crisis_start": "2007-08-01",
        "crisis_peak": "2008-10-24",
        "crisis_end": "2009-06-30",
        "zoom_start": "2007-01-01",
        "zoom_end": "2009-12-31",
        "post_start": "2009-07-01",
        "post_end": "2009-12-31",
    },
    "Subprime_SP500": {
        "file": "S&P 500 과거 데이터.csv",
        "title": "S&P 500 during the Global Financial Crisis",
        "crisis_start": "2008-09-01",
        "crisis_peak": "2009-03-09",
        "crisis_end": "2009-06-30",
        "zoom_start": "2007-01-01", 
        "zoom_end": "2009-12-31",  
        "post_start": "2009-07-01",
        "post_end": "2009-09-30",
    },
    "COVID_KOSPI": {
        "file": "COVID_KOSPI.csv",
        "title": "KOSPI during the COVID-19 shock",
        "crisis_start": "2020-02-20",
        "crisis_peak": "2020-03-19",
        "crisis_end": "2020-05-31",
        "zoom_start": "2019-07-01",
        "zoom_end": "2020-10-31",
        "post_start": "2020-06-01",
        "post_end": "2020-10-31",
    },
    "COVID_SP500": {
        "file": "COVID_SP_500.csv",
        "title": "S&P 500 during the COVID-19 shock",
        "crisis_start": "2020-02-20",
        "crisis_peak": "2020-03-23",
        "crisis_end": "2020-05-31",
        "zoom_start": "2019-07-01",
        "zoom_end": "2020-10-31",
        "post_start": "2020-06-01",
        "post_end": "2020-10-31",
    },
}

ROW_ORDER = ["Subprime_KOSPI", "Subprime_SP500", "COVID_KOSPI", "COVID_SP500"]


# Utilities
def clean_date_column(x):
    s = str(x).strip()
    s = s.replace(" ", "")
    return pd.to_datetime(s, errors="coerce")

def load_market_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["날짜"] = df["날짜"].apply(clean_date_column)

    for col in ["종가", "시가", "고가", "저가"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )

    df = df.dropna(subset=["날짜", "종가"]).sort_values("날짜").reset_index(drop=True)

    df["log_price"] = np.log(df["종가"])
    df["log_return"] = df["log_price"].diff()
    df["ret_pct"] = 100.0 * df["log_return"]
    df["ret_dec"] = df["log_return"]

    df["sq_ret"] = df["ret_pct"] ** 2
    df["rv_20"] = df["ret_pct"].rolling(20).std()
    df["rv_60"] = df["ret_pct"].rolling(60).std()

    return df

def build_simple_rdp(df):
    """
    Simple regime-dependent persistence approximation for supplementary figures.
    """
    work = df[["날짜", "ret_dec"]].copy().dropna().reset_index(drop=True)
    r = work["ret_dec"].to_numpy()
    T = len(r)

    threshold = np.quantile(np.abs(r), 0.90)
    high_state = (np.abs(r) > threshold).astype(int)

    p_high = pd.Series(high_state).rolling(10, min_periods=1).mean().to_numpy()

    alpha_low, beta_low = 0.08, 0.50
    alpha_high, beta_high = 0.10, 0.88

    rho_low = alpha_low + beta_low
    rho_high = alpha_high + beta_high

    var_r = np.var(r)
    omega_low = max(var_r * (1.0 - rho_low), 1e-10)
    omega_high = max((4.0 * var_r) * (1.0 - rho_high), 1e-10)

    h_rdp = np.zeros(T)
    h_rdp[0] = max(var_r, 1e-10)

    for t in range(1, T):
        if high_state[t] == 1:
            omega_t, alpha_t, beta_t = omega_high, alpha_high, beta_high
        else:
            omega_t, alpha_t, beta_t = omega_low, alpha_low, beta_low

        h_rdp[t] = omega_t + alpha_t * (r[t - 1] ** 2) + beta_t * h_rdp[t - 1]
        h_rdp[t] = max(h_rdp[t], 1e-10)

    out = pd.DataFrame({
        "date": work["날짜"],
        "h_rdp": h_rdp * (100.0 ** 2),
        "p_high": p_high,
        "rho_low": rho_low,
        "rho_high": rho_high
    })
    return out

def half_life(rho):
    if pd.isna(rho):
        return np.nan
    if rho <= 0:
        return 0.0
    if rho >= 1:
        return np.inf
    return np.log(0.5) / np.log(rho)

def apply_date_format(ax, interval=3):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)


# Load all cases
all_results = {}

for key, meta in CASES.items():
    csv_path = os.path.join(CODE_DIR, meta["file"])
    df = load_market_csv(csv_path)
    rdp = build_simple_rdp(df)

    df = df.merge(rdp, left_on="날짜", right_on="date", how="left")
    df["vol_rdp"] = np.sqrt(df["h_rdp"])

    all_results[key] = {
        "df": df,
        "meta": meta,
        "rho_low": float(df["rho_low"].dropna().iloc[0]),
        "rho_high": float(df["rho_high"].dropna().iloc[0]),
    }


# Plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Figure S1: Crisis dynamics
fig, axes = plt.subplots(4, 3, figsize=(18, 18), sharex=False)

for i, key in enumerate(ROW_ORDER):
    df = all_results[key]["df"]
    meta = all_results[key]["meta"]

    crisis_start = pd.Timestamp(meta["crisis_start"])
    crisis_peak = pd.Timestamp(meta["crisis_peak"])
    crisis_end = pd.Timestamp(meta["crisis_end"])
    zoom_start = pd.Timestamp(meta["zoom_start"])
    zoom_end = pd.Timestamp(meta["zoom_end"])

    sub = df[(df["날짜"] >= zoom_start) & (df["날짜"] <= zoom_end)].copy()

    # Index
    ax = axes[i, 0]
    ax.plot(sub["날짜"], sub["종가"], color="#3A7D44")
    ax.axvspan(crisis_start, crisis_end, color="#d9d9d9", alpha=0.30)
    ax.axvline(crisis_peak, color="#333333", linestyle="--", linewidth=1)
    ax.set_title(f"{meta['title']}\nIndex")
    ax.set_ylabel("Index")
    apply_date_format(ax, interval=3)

    # Returns
    ax = axes[i, 1]
    ax.plot(sub["날짜"], sub["ret_pct"], color="black", linewidth=0.9)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.axvspan(crisis_start, crisis_end, color="#d9d9d9", alpha=0.30)
    ax.axvline(crisis_peak, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Returns")
    ax.set_ylabel("Percent")
    apply_date_format(ax, interval=3)

    # Volatility
    ax = axes[i, 2]
    ax.plot(sub["날짜"], sub["rv_20"], color="#d62728", linestyle=":", label="20-day rolling volatility")
    ax.plot(sub["날짜"], sub["vol_rdp"], color="#1f77b4", label="RDP-GARCH")
    ax.axvspan(crisis_start, crisis_end, color="#d9d9d9", alpha=0.30)
    ax.axvline(crisis_peak, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Volatility")
    ax.set_ylabel("Volatility")
    apply_date_format(ax, interval=3)
    if i == 0:
        ax.legend(frameon=False, loc="upper left")

for j in range(3):
    axes[-1, j].set_xlabel("Date")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_S1.eps"), format="eps", bbox_inches="tight")
plt.close()


# Figure S2: Post-crisis adjustment
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
axes = axes.flatten()

for idx, key in enumerate(ROW_ORDER):
    ax = axes[idx]
    df = all_results[key]["df"]
    meta = all_results[key]["meta"]

    post_start = pd.Timestamp(meta["post_start"])
    post_end = pd.Timestamp(meta["post_end"])

    sub = df[(df["날짜"] >= post_start) & (df["날짜"] <= post_end)].copy()

    ax.plot(sub["날짜"], sub["rv_20"], color="#d62728", linestyle=":", label="20-day rolling volatility")
    ax.plot(sub["날짜"], sub["vol_rdp"], color="#1f77b4", label="RDP-GARCH")
    ax.set_title(meta["title"])
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")
    apply_date_format(ax, interval=2)
    ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_S2.eps"), format="eps", bbox_inches="tight")
plt.close()


# Figure S3: Persistence comparison
labels = []
low_vals = []
high_vals = []

for key in ROW_ORDER:
    meta = all_results[key]["meta"]
    labels.append(meta["title"].replace(" during the ", "\n"))
    low_vals.append(all_results[key]["rho_low"])
    high_vals.append(all_results[key]["rho_high"])

x = np.arange(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(16, 7))
bars1 = ax.bar(x - width / 2, low_vals, width, color="#D9D9D9", edgecolor="black", label="Low regime")
bars2 = ax.bar(x + width / 2, high_vals, width, color="#404040", edgecolor="black", label="High regime")

ax.set_ylabel("Persistence")
ax.set_title("Persistence comparison across markets and crisis episodes")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(frameon=False, loc="upper left")

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_S3.eps"), format="eps", bbox_inches="tight")
plt.close()


# Figure S4: Regime probability
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
axes = axes.flatten()

for idx, key in enumerate(ROW_ORDER):
    ax = axes[idx]
    df = all_results[key]["df"]
    meta = all_results[key]["meta"]

    crisis_start = pd.Timestamp(meta["crisis_start"])
    crisis_peak = pd.Timestamp(meta["crisis_peak"])
    crisis_end = pd.Timestamp(meta["crisis_end"])
    zoom_start = pd.Timestamp(meta["zoom_start"])
    zoom_end = pd.Timestamp(meta["zoom_end"])

    sub = df[(df["날짜"] >= zoom_start) & (df["날짜"] <= zoom_end)].copy()

    ax.plot(sub["날짜"], sub["p_high"], color="#1f77b4")
    ax.axvspan(crisis_start, crisis_end, color="#d9d9d9", alpha=0.30)
    ax.axvline(crisis_peak, color="#333333", linestyle="--", linewidth=1)
    ax.set_title(meta["title"])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    apply_date_format(ax, interval=3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_S4.eps"), format="eps", bbox_inches="tight")
plt.close()


# Optional: summary table
summary_rows = []
for key in ROW_ORDER:
    meta = all_results[key]["meta"]
    rho_low = all_results[key]["rho_low"]
    rho_high = all_results[key]["rho_high"]

    summary_rows.append([
        meta["title"],
        rho_low,
        half_life(rho_low),
        rho_high,
        half_life(rho_high)
    ])

summary = pd.DataFrame(
    summary_rows,
    columns=["Case", "rho_low", "half_life_low", "rho_high", "half_life_high"]
)

summary.to_csv(
    os.path.join(DATA_DIR, "supplement_persistence_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("Saved supplementary figures:")
