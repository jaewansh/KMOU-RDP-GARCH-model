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
FIG_DIR  = os.path.join(DATA_DIR, "figure")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PRICE_CSV = os.path.join(CODE_DIR, "96-99(IMF)_KOSPI.csv")
RDP_CSV   = os.path.join(DATA_DIR, "IMF_KOSPI_rdp_results.csv")


# Journal-style color palette
COLOR_RDP   = "#1f77b4"   # blue
COLOR_STD   = "#d62728"   # red
COLOR_ROLL  = "#7f7f7f"   # gray
COLOR_SHADE = "#d9d9d9"   # light gray


# Common style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# Helper: date formatting
def style_date_axis(ax, month_interval=4, rotate=45):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=rotate)


# 1. Load market data
df = pd.read_csv(PRICE_CSV)
df.columns = [c.strip() for c in df.columns]

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["종가"] = pd.to_numeric(
    df["종가"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce"
)

df = df.dropna(subset=["날짜", "종가"]).sort_values("날짜").reset_index(drop=True)

df["log_price"] = np.log(df["종가"])
df["log_return"] = df["log_price"].diff()
df["ret_pct"] = 100.0 * df["log_return"]
df["ret_dec"] = df["log_return"]

# realized volatility proxy
df["sq_ret"] = df["ret_pct"] ** 2
df["rv_20"] = df["ret_pct"].rolling(20).std()
df["rv_60"] = df["ret_pct"].rolling(60).std()


# 2. Crisis windows
crisis_start = pd.Timestamp("1997-07-01")
crisis_peak  = pd.Timestamp("1997-11-21")
crisis_end   = pd.Timestamp("1998-12-31")

post_start = crisis_end + pd.Timedelta(days=1)
post_end   = pd.Timestamp("1999-12-31")

zoom_start = pd.Timestamp("1997-01-01")
zoom_end   = pd.Timestamp("1999-12-31")


# 3. Standard GARCH fit
garch_ok = False
alpha_std = np.nan
beta_std = np.nan
rho_std = np.nan

try:
    from arch import arch_model

    ret_series = df["ret_pct"].dropna()
    am = arch_model(ret_series, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    df["h_std"] = np.nan
    df.loc[ret_series.index, "h_std"] = res.conditional_volatility.values ** 2
    df["vol_std"] = np.sqrt(df["h_std"])

    alpha_std = float(res.params.get("alpha[1]", np.nan))
    beta_std  = float(res.params.get("beta[1]", np.nan))
    rho_std   = alpha_std + beta_std

    garch_ok = True
    print("Standard GARCH fitted.")
    print(f"alpha={alpha_std:.4f}, beta={beta_std:.4f}, rho={rho_std:.4f}")

except Exception as e:
    print("Standard GARCH fitting failed:", e)
    df["h_std"] = np.nan
    df["vol_std"] = np.nan

# ============================================================
# 4. Build RDP results automatically if missing
#    This is a simple regime-dependent approximation.
# ============================================================
def build_rdp_results(input_df, save_csv):
    work = input_df[["날짜", "ret_dec"]].copy().dropna().reset_index(drop=True)
    r = work["ret_dec"].to_numpy()
    T = len(r)

    # state proxy: large absolute returns -> high-vol regime
    threshold = np.quantile(np.abs(r), 0.90)
    high_state = (np.abs(r) > threshold).astype(int)

    # smooth state probability
    p_high = pd.Series(high_state).rolling(10, min_periods=1).mean().to_numpy()

    # regime-specific persistence
    alpha_low, beta_low = 0.08, 0.50
    alpha_high, beta_high = 0.10, 0.88

    rho_low = alpha_low + beta_low
    rho_high = alpha_high + beta_high

    # keep low-regime unconditional variance tied to sample scale
    var_r = np.var(r)
    omega_low = max(var_r * (1.0 - rho_low), 1e-8)
    omega_high = max((4.0 * var_r) * (1.0 - rho_high), 1e-8)

    h_rdp = np.zeros(T)
    h_rdp[0] = max(var_r, 1e-8)

    for t in range(1, T):
        if high_state[t] == 1:
            omega_t, alpha_t, beta_t = omega_high, alpha_high, beta_high
        else:
            omega_t, alpha_t, beta_t = omega_low, alpha_low, beta_low

        h_rdp[t] = omega_t + alpha_t * (r[t-1] ** 2) + beta_t * h_rdp[t-1]
        h_rdp[t] = max(h_rdp[t], 1e-10)

    out = pd.DataFrame({
        "date": work["날짜"],
        # convert to percent-squared scale to match h_std and sq_ret
        "h_rdp": h_rdp * (100.0 ** 2),
        "p_high": p_high,
        "rho_low": rho_low,
        "rho_high": rho_high
    })

    out.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"RDP results created: {save_csv}")
    return out

if not os.path.exists(RDP_CSV):
    build_rdp_results(df, RDP_CSV)


# 5. Load RDP results
rdp_ok = False
rho_low = np.nan
rho_high = np.nan

if os.path.exists(RDP_CSV):
    rdp = pd.read_csv(RDP_CSV)
    rdp.columns = [c.strip() for c in rdp.columns]

    if "date" not in rdp.columns or "h_rdp" not in rdp.columns:
        raise ValueError("RDP result file must contain at least 'date' and 'h_rdp' columns.")

    rdp["date"] = pd.to_datetime(rdp["date"], errors="coerce")
    df = df.merge(rdp, left_on="날짜", right_on="date", how="left")

    df["vol_rdp"] = np.sqrt(df["h_rdp"])

    if "rho_low" in df.columns and df["rho_low"].notna().any():
        rho_low = float(df["rho_low"].dropna().iloc[0])

    if "rho_high" in df.columns and df["rho_high"].notna().any():
        rho_high = float(df["rho_high"].dropna().iloc[0])

    rdp_ok = True
    print("RDP results loaded.")
else:
    df["h_rdp"] = np.nan
    df["vol_rdp"] = np.nan
    df["p_high"] = np.nan


# 6. Helper functions
def qlike_proxy(h_hat, y_proxy):
    h_hat = np.maximum(h_hat, 1e-10)
    y_proxy = np.maximum(y_proxy, 1e-10)
    return np.log(h_hat) + y_proxy / h_hat

def recovery_time(series_hat, series_true, tol=0.10):
    valid = series_hat.notna() & series_true.notna()
    if valid.sum() == 0:
        return np.nan

    x = series_hat[valid].to_numpy()
    y = series_true[valid].to_numpy()
    rel = (x - y) / np.maximum(y, 1e-10)

    for i in range(len(rel)):
        if np.all(np.abs(rel[i:]) <= tol):
            return i
    return len(rel)

def mse(x, y):
    v = pd.concat([x, y], axis=1).dropna()
    if len(v) == 0:
        return np.nan
    return np.mean((v.iloc[:, 0] - v.iloc[:, 1]) ** 2)

def mae(x, y):
    v = pd.concat([x, y], axis=1).dropna()
    if len(v) == 0:
        return np.nan
    return np.mean(np.abs(v.iloc[:, 0] - v.iloc[:, 1]))

def mean_qlike(x, y):
    v = pd.concat([x, y], axis=1).dropna()
    if len(v) == 0:
        return np.nan
    return np.mean(qlike_proxy(v.iloc[:, 0], v.iloc[:, 1]))

def half_life(rho):
    if pd.isna(rho):
        return np.nan
    if rho <= 0:
        return 0.0
    if rho >= 1:
        return np.inf
    return np.log(0.5) / np.log(rho)


# 7. Crisis case study: Figure C1
zoom_df = df[(df["날짜"] >= zoom_start) & (df["날짜"] <= zoom_end)].copy()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# (a) Index
ax = axes[0]
ax.plot(zoom_df["날짜"], zoom_df["종가"], color="black", linewidth=2.0)
ax.axvspan(crisis_start, crisis_end, color=COLOR_SHADE, alpha=0.30)
ax.axvline(crisis_peak, color="black", linestyle="--", linewidth=1.0)
ax.set_title("(a) KOSPI index around the IMF crisis")
ax.set_ylabel("Index")

# (b) Returns
ax = axes[1]
ax.plot(zoom_df["날짜"], zoom_df["ret_pct"], color="black", linewidth=1.0)
ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
ax.axvspan(crisis_start, crisis_end, color=COLOR_SHADE, alpha=0.30)
ax.axvline(crisis_peak, color="black", linestyle="--", linewidth=1.0)
ax.set_title("(b) Daily returns")
ax.set_ylabel("Percent")

# (c) Volatility
ax = axes[2]
ax.plot(
    zoom_df["날짜"], zoom_df["rv_20"],
    color=COLOR_ROLL, linestyle=":", linewidth=2.0,
    label="20-day rolling volatility"
)
if garch_ok:
    ax.plot(
        zoom_df["날짜"], zoom_df["vol_std"],
        color=COLOR_STD, linestyle="--", linewidth=2.0,
        label="Standard GARCH"
    )
if rdp_ok:
    ax.plot(
        zoom_df["날짜"], zoom_df["vol_rdp"],
        color=COLOR_RDP, linestyle="-", linewidth=2.2,
        label="RDP-GARCH"
    )

ax.axvspan(crisis_start, crisis_end, color=COLOR_SHADE, alpha=0.30)
ax.axvline(crisis_peak, color="black", linestyle="--", linewidth=1.0)
ax.set_title("(c) Volatility estimates")
ax.set_ylabel("Volatility")
ax.set_xlabel("Date")
ax.legend(frameon=False, loc="upper left")

style_date_axis(ax, month_interval=4, rotate=45)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_C1.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Figure_C1.eps"), format="eps", bbox_inches="tight")
plt.close()


# 8. Post-crisis overestimation comparison: Figure C2
post_df = df[(df["날짜"] >= post_start) & (df["날짜"] <= post_end)].copy()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    post_df["날짜"], post_df["rv_20"],
    color=COLOR_ROLL, linestyle=":", linewidth=2.0,
    label="20-day rolling volatility"
)

if garch_ok:
    ax.plot(
        post_df["날짜"], post_df["vol_std"],
        color=COLOR_STD, linestyle="--", linewidth=2.0,
        label="Standard GARCH"
    )

if rdp_ok:
    ax.plot(
        post_df["날짜"], post_df["vol_rdp"],
        color=COLOR_RDP, linestyle="-", linewidth=2.2,
        label="RDP-GARCH"
    )

ax.set_title("Post-crisis volatility adjustment")
ax.set_ylabel("Volatility")
ax.set_xlabel("Date")
ax.legend(frameon=False, loc="upper left")

style_date_axis(ax, month_interval=2, rotate=45)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure_C2.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Figure_C2.eps"), format="eps", bbox_inches="tight")
plt.close()

summary_rows = []
target = post_df["sq_ret"]

if garch_ok:
    over_std = post_df["h_std"] / np.maximum(target, 1e-10)
    summary_rows.append([
        "Standard GARCH",
        np.nanmean(over_std),
        np.nansum(np.maximum(post_df["h_std"] - target, 0)),
        recovery_time(post_df["h_std"], target)
    ])

if rdp_ok:
    over_rdp = post_df["h_rdp"] / np.maximum(target, 1e-10)
    summary_rows.append([
        "RDP-GARCH",
        np.nanmean(over_rdp),
        np.nansum(np.maximum(post_df["h_rdp"] - target, 0)),
        recovery_time(post_df["h_rdp"], target)
    ])

over_table = pd.DataFrame(
    summary_rows,
    columns=["Model", "Average OR proxy", "Cumulative overestimation", "Recovery time"]
)
over_table.to_csv(
    os.path.join(DATA_DIR, "Empirical_Table_post_crisis_overestimation.csv"),
    index=False,
    encoding="utf-8-sig"
)


# 9. Forecast performance table
perf_rows = []

periods = {
    "Full sample": df.index,
    "Crisis": df[(df["날짜"] >= crisis_start) & (df["날짜"] <= crisis_end)].index,
    "Post-crisis": df[(df["날짜"] >= post_start) & (df["날짜"] <= post_end)].index
}

for period_name, idx in periods.items():
    y = df.loc[idx, "sq_ret"]

    if garch_ok:
        x = df.loc[idx, "h_std"]
        perf_rows.append([
            period_name, "Standard GARCH",
            mean_qlike(x, y),
            mse(x, y),
            mae(x, y)
        ])

    if rdp_ok:
        x = df.loc[idx, "h_rdp"]
        perf_rows.append([
            period_name, "RDP-GARCH",
            mean_qlike(x, y),
            mse(x, y),
            mae(x, y)
        ])

perf_table = pd.DataFrame(
    perf_rows,
    columns=["Period", "Model", "QLIKE", "MSE", "MAE"]
)
perf_table.to_csv(
    os.path.join(DATA_DIR, "Empirical_Table_forecast_performance.csv"),
    index=False,
    encoding="utf-8-sig"
)


# 10. Persistence analysis: Figure C3
persist_rows = []

if garch_ok:
    persist_rows.append([
        "Standard GARCH",
        alpha_std,
        beta_std,
        rho_std,
        half_life(rho_std)
    ])

if rdp_ok:
    persist_rows.append([
        "RDP-GARCH (low regime)",
        np.nan, np.nan,
        rho_low,
        half_life(rho_low)
    ])
    persist_rows.append([
        "RDP-GARCH (high regime)",
        np.nan, np.nan,
        rho_high,
        half_life(rho_high)
    ])

persist_table = pd.DataFrame(
    persist_rows,
    columns=["Model/Regime", "alpha", "beta", "Persistence (rho)", "Half-life"]
)
persist_table.to_csv(
    os.path.join(DATA_DIR, "Empirical_Table_persistence_analysis.csv"),
    index=False,
    encoding="utf-8-sig"
)

if len(persist_table) > 0:
    labels = persist_table["Model/Regime"].tolist()
    vals = persist_table["Persistence (rho)"].tolist()

    colors = []
    for label in labels:
        if "Standard" in label:
            colors.append(COLOR_STD)
        else:
            colors.append(COLOR_RDP)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        np.arange(len(labels)),
        vals,
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Persistence")
    ax.set_title("Persistence comparison across models/regimes")

    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_C3.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "Figure_C3.eps"), format="eps", bbox_inches="tight")
    plt.close()


# 11. Regime probability figure: Figure C4
if rdp_ok and "p_high" in df.columns:
    prob_df = df[(df["날짜"] >= zoom_start) & (df["날짜"] <= zoom_end)].copy()

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(prob_df["날짜"], prob_df["p_high"], color=COLOR_RDP, linewidth=2.2)
    ax.axvspan(crisis_start, crisis_end, color=COLOR_SHADE, alpha=0.30)
    ax.axvline(crisis_peak, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("High-volatility regime probability")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")

    style_date_axis(ax, month_interval=4, rotate=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_C4.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "Figure_C4.eps"), format="eps", bbox_inches="tight")
    plt.close()


# 12. Save merged data
df.to_csv(
    os.path.join(DATA_DIR, "IMF_KOSPI_empirical_merged.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved outputs:")
