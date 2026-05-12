import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LANG_COLORS = {"ko": "#4C72B0", "ja": "#DD8452", "zh": "#55A868", "en": "#C44E52"}
LANG_ORDER = ["ko", "ja", "zh", "en"]
LANG_LABELS = {"ko": "Korean", "ja": "Japanese", "zh": "Chinese", "en": "English"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

BASE = os.path.dirname(__file__)

def load(name):
    return pd.read_csv(os.path.join(BASE, name), encoding="utf-8-sig")

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── 1. Language Overview ──────────────────────────────────────────────────────
df = load("01_language_overview.csv")
df = df[df["language"] != "total"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([LANG_LABELS[l] for l in LANG_ORDER],
              [df.loc[df.language==l, "model_count"].values[0] for l in LANG_ORDER],
              color=[LANG_COLORS[l] for l in LANG_ORDER], width=0.5, edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{bar.get_height():,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Model Count by Language", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Models")
ax.set_ylim(0, 18500)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "01_language_overview.png")

# ── 2. Pipeline Tag Distribution (Top 12) ────────────────────────────────────
df = load("02_pipeline_tag_by_lang.csv")
df = df[df["pipeline_tag"] != "None"].nlargest(12, "total")

x = np.arange(len(df))
w = 0.2
fig, ax = plt.subplots(figsize=(13, 5))
for i, lang in enumerate(LANG_ORDER):
    ax.bar(x + i*w, df[lang], w, label=LANG_LABELS[lang],
           color=LANG_COLORS[lang], edgecolor="white")
ax.set_title("Top 12 Pipeline Tags by Language", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df["pipeline_tag"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "02_pipeline_tag.png")

# ── 3. Library Distribution (Top 10, stacked) ────────────────────────────────
df = load("03_library_by_lang.csv")
df = df[df["library_name"] != "None"].nlargest(10, "total")

fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(df))
lib_labels = df["library_name"].astype(str).tolist()
for lang in LANG_ORDER:
    vals = df[lang].values.astype(float)
    bars = ax.bar(lib_labels, vals, bottom=bottom,
                  label=LANG_LABELS[lang], color=LANG_COLORS[lang], edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_title("Top 10 ML Libraries (Stacked by Language)", fontsize=14, fontweight="bold", pad=12)
ax.set_xticklabels(lib_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "03_library.png")

# ── 4. License Distribution (Top 10) ─────────────────────────────────────────
df = load("04_license_by_lang.csv")
df = df[~df["license"].isin(["N/A", "other", "unknown"])].nlargest(10, "total")

x = np.arange(len(df))
w = 0.2
fig, ax = plt.subplots(figsize=(12, 5))
for i, lang in enumerate(LANG_ORDER):
    ax.bar(x + i*w, df[lang], w, label=LANG_LABELS[lang],
           color=LANG_COLORS[lang], edgecolor="white")
ax.set_title("Top 10 Licenses by Language", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df["license"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "04_license.png")

# ── 5. Org vs Individual ──────────────────────────────────────────────────────
df = load("05_org_vs_user_by_lang.csv")
df = df[df["language"] != "total"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Stacked bar
ax = axes[0]
langs = [LANG_LABELS[l] for l in LANG_ORDER]
orgs  = [df.loc[df.language==l, "organization"].values[0] for l in LANG_ORDER]
indvs = [df.loc[df.language==l, "individual"].values[0] for l in LANG_ORDER]
ax.bar(langs, orgs,  label="Organization", color="#4C72B0", edgecolor="white")
ax.bar(langs, indvs, bottom=orgs, label="Individual", color="#C44E52", edgecolor="white")
for i, (o, ind) in enumerate(zip(orgs, indvs)):
    pct = o / (o + ind) * 100
    ax.text(i, o + ind + 100, f"Org {pct:.1f}%", ha="center", fontsize=9, color="#333")
ax.set_title("Organization vs Individual Authors", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(framealpha=0.9)

# Org ratio bar
ax2 = axes[1]
ratios = [df.loc[df.language==l, "org_ratio(%)"].values[0] for l in LANG_ORDER]
bars = ax2.bar(langs, ratios, color=[LANG_COLORS[l] for l in LANG_ORDER], edgecolor="white", width=0.5)
for bar, r in zip(bars, ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Organization Ratio (%)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Org Ratio (%)")
ax2.set_ylim(0, 75)
plt.tight_layout()
save(fig, "05_org_vs_individual.png")

# ── 6. Yearly Growth ──────────────────────────────────────────────────────────
df = load("06_yearly_growth.csv")
df = df[df["year"] != 2026]  # 2026은 진행 중

linestyles = {"ko": "-", "ja": "--", "zh": "-.", "en": ":"}
markers    = {"ko": "o", "ja": "s", "zh": "^", "en": "D"}
fig, ax = plt.subplots(figsize=(8, 4))
for lang in LANG_ORDER:
    ax.plot(df["year"].astype(str), df[lang],
            marker=markers[lang], linestyle=linestyles[lang],
            label=LANG_LABELS[lang], color=LANG_COLORS[lang], linewidth=2, markersize=6)
    for y, v in zip(df["year"].astype(str), df[lang]):
        ax.text(y, v + 40, f"{v:,}", ha="center", fontsize=7.5, color=LANG_COLORS[lang])
ax.set_title("New Models per Year (2022–2025)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("New Models Added")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "06_yearly_growth.png")

# ── 7. Parameter Size Distribution ───────────────────────────────────────────
df = load("07_param_size_distribution.csv")
df = df[df["param_size"] != "unknown"]

x = np.arange(len(df))
w = 0.2
fig, ax = plt.subplots(figsize=(10, 4))
for i, lang in enumerate(LANG_ORDER):
    ax.bar(x + i*w, df[lang], w, label=LANG_LABELS[lang],
           color=LANG_COLORS[lang], edgecolor="white")
ax.set_title("Parameter Size Distribution (excluding unknown)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df["param_size"], rotation=20, ha="right")
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "07_param_size.png")

# ── 8. Top 10 Authors (ko only, most representative) ─────────────────────────
df = load("08_top_authors_by_lang.csv")
df_ko = df[df["language"] == "ko"].head(10)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#4C72B0" if o else "#DD8452" for o in df_ko["is_organization"]]
bars = ax.barh(df_ko["author"][::-1], df_ko["model_count"][::-1], color=colors[::-1], edgecolor="white")
for bar in bars:
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width()):,}", va="center", fontsize=9)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#4C72B0", label="Organization"),
                   Patch(facecolor="#DD8452", label="Individual")]
ax.legend(handles=legend_elements, framealpha=0.9)
ax.set_title("Top 10 Authors (Korean models)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Number of Models")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "08_top_authors_ko.png")

# ── 9. Gated / Private / Disabled ────────────────────────────────────────────
df = load("09_gated_private_disabled.csv")
df = df[df["language"] != "total"]

cols = ["gated(%)", "private(%)", "disabled(%)"]
labels = ["Gated", "Private", "Disabled"]
x = np.arange(len(LANG_ORDER))
w = 0.25

fig, ax = plt.subplots(figsize=(8, 4))
for i, (col, label) in enumerate(zip(cols, labels)):
    vals = [df.loc[df.language==l, col].values[0] for l in LANG_ORDER]
    bars = ax.bar(x + i*w, vals, w, label=label, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{v:.1f}%", ha="center", fontsize=8)
ax.set_title("Gated / Private / Disabled Rate by Language", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x + w)
ax.set_xticklabels([LANG_LABELS[l] for l in LANG_ORDER])
ax.set_ylabel("Rate (%)")
ax.legend(framealpha=0.9)
save(fig, "09_gated_private_disabled.png")

# ── 10. Model Tree Summary ────────────────────────────────────────────────────
df = load("10_model_tree_summary.csv")

tree_cols = [("adapters_mean","Adapters"), ("finetunes_mean","Finetunes"),
             ("quantizations_mean","Quantizations"), ("merges_mean","Merges")]
x = np.arange(len(tree_cols))
w = 0.2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Mean derivatives per model
ax = axes[0]
for i, lang in enumerate(LANG_ORDER):
    row = df[df.language == lang].iloc[0]
    vals = [row[c] for c, _ in tree_cols]
    ax.bar(x + i*w, vals, w, label=LANG_LABELS[lang],
           color=LANG_COLORS[lang], edgecolor="white")
ax.set_title("Mean Derivative Count per Model", fontsize=12, fontweight="bold")
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([l for _, l in tree_cols])
ax.set_ylabel("Mean Count")
ax.legend(title="Language", framealpha=0.9)

# Total
total_cols = [("adapters_total","Adapters"), ("finetunes_total","Finetunes"),
              ("quantizations_total","Quantizations"), ("merges_total","Merges")]
ax2 = axes[1]
for i, lang in enumerate(LANG_ORDER):
    row = df[df.language == lang].iloc[0]
    vals = [row[c] for c, _ in total_cols]
    ax2.bar(x + i*w, vals, w, label=LANG_LABELS[lang],
            color=LANG_COLORS[lang], edgecolor="white")
ax2.set_title("Total Derivative Count", fontsize=12, fontweight="bold")
ax2.set_xticks(x + w*1.5)
ax2.set_xticklabels([l for _, l in total_cols])
ax2.set_ylabel("Total Count")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1e6):.1f}M" if x >= 1e6 else f"{int(x/1e3)}K"))
ax2.legend(title="Language", framealpha=0.9)

plt.tight_layout()
save(fig, "10_model_tree.png")

# ── 11. Monthly Growth ────────────────────────────────────────────────────────
df = load("11_monthly_growth.csv")
df["date"] = pd.to_datetime(df["year_month"])
df = df.sort_values("date")

import matplotlib.dates as mdates
linestyles = {"ko": "-", "ja": "--", "zh": "-.", "en": ":"}
fig, ax = plt.subplots(figsize=(14, 5))
for lang in LANG_ORDER:
    ax.plot(df["date"], df[lang],
            label=LANG_LABELS[lang], color=LANG_COLORS[lang],
            linestyle=linestyles[lang], linewidth=1.8, alpha=0.9)
# 주요 이벤트 annotate
ax.axvline(pd.Timestamp("2023-01-01"), color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(pd.Timestamp("2023-01-15"), ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 900,
        "2023", fontsize=8, color="gray")
ax.axvline(pd.Timestamp("2025-07-01"), color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(pd.Timestamp("2025-07-15"), 1000, "2025-07\nSpike", fontsize=7.5, color="gray")
ax.set_title("Monthly New Model Additions (Mar 2022 – Apr 2026)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("New Models per Month")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30, ha="right")
save(fig, "11_monthly_growth.png")

# ── 12. Param Source Distribution ────────────────────────────────────────────
df = load("12_param_source_distribution.csv")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(df))
w = 0.2
for i, lang in enumerate(LANG_ORDER):
    ax.bar(x + i*w, df[lang], w, label=LANG_LABELS[lang],
           color=LANG_COLORS[lang], edgecolor="white")
ax.set_title("Parameter Count Source Distribution", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df["param_source"])
ax.set_ylabel("Number of Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Language", framealpha=0.9)
save(fig, "12_param_source.png")

# ── 13. arXiv Coverage ───────────────────────────────────────────────────────
df = load("13_arxiv_coverage.csv")
df = df[df["language"] != "total"]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(LANG_ORDER))
w = 0.35
has = [df.loc[df.language==l, "has_arxiv(%)"].values[0] for l in LANG_ORDER]
no  = [df.loc[df.language==l, "no_arxiv(%)"].values[0] for l in LANG_ORDER]
ax.bar(x, has, w, label="Has arXiv", color="#4C72B0", edgecolor="white")
ax.bar(x, no, w, bottom=has, label="No arXiv", color="#CCCCCC", edgecolor="white")
for i, (h, n) in enumerate(zip(has, no)):
    ax.text(i, h/2, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold", color="white")
ax.set_title("arXiv Paper Coverage by Language", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([LANG_LABELS[l] for l in LANG_ORDER])
ax.set_ylabel("Percentage (%)")
ax.legend(framealpha=0.9)
save(fig, "13_arxiv_coverage.png")

# ── Combined Overview (2x2 highlight) ────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# [0,0] Model count
ax = fig.add_subplot(gs[0, 0])
df01 = load("01_language_overview.csv")
df01 = df01[df01["language"] != "total"]
bars = ax.bar([LANG_LABELS[l] for l in LANG_ORDER],
              [df01.loc[df01.language==l, "model_count"].values[0] for l in LANG_ORDER],
              color=[LANG_COLORS[l] for l in LANG_ORDER], edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{bar.get_height():,}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("① Model Count", fontweight="bold")
ax.set_ylim(0, 19000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# [0,1] Org ratio
ax = fig.add_subplot(gs[0, 1])
df05 = load("05_org_vs_user_by_lang.csv")
df05 = df05[df05["language"] != "total"]
ratios = [df05.loc[df05.language==l, "org_ratio(%)"].values[0] for l in LANG_ORDER]
bars = ax.bar([LANG_LABELS[l] for l in LANG_ORDER], ratios,
              color=[LANG_COLORS[l] for l in LANG_ORDER], edgecolor="white", width=0.5)
for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("② Organization Ratio", fontweight="bold")
ax.set_ylim(0, 75)
ax.set_ylabel("%")

# [1,0] Yearly growth
ax = fig.add_subplot(gs[1, 0])
df06 = load("06_yearly_growth.csv")
df06 = df06[df06["year"] != 2026]
linestyles = {"ko": "-", "ja": "--", "zh": "-.", "en": ":"}
markers    = {"ko": "o", "ja": "s", "zh": "^", "en": "D"}
for lang in LANG_ORDER:
    ax.plot(df06["year"].astype(str), df06[lang],
            marker=markers[lang], linestyle=linestyles[lang],
            label=LANG_LABELS[lang], color=LANG_COLORS[lang], linewidth=2, markersize=6)
ax.set_title("③ Yearly Growth", fontweight="bold")
ax.set_ylabel("New Models")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, framealpha=0.9)

# [1,1] arXiv coverage
ax = fig.add_subplot(gs[1, 1])
df13 = load("13_arxiv_coverage.csv")
df13 = df13[df13["language"] != "total"]
has = [df13.loc[df13.language==l, "has_arxiv(%)"].values[0] for l in LANG_ORDER]
no  = [df13.loc[df13.language==l, "no_arxiv(%)"].values[0] for l in LANG_ORDER]
ax.bar([LANG_LABELS[l] for l in LANG_ORDER], has, 0.5, label="Has arXiv", color="#4C72B0", edgecolor="white")
ax.bar([LANG_LABELS[l] for l in LANG_ORDER], no, 0.5, bottom=has, label="No arXiv", color="#CCCCCC", edgecolor="white")
for i, h in enumerate(has):
    ax.text(i, h/2, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold", color="white")
ax.set_title("④ arXiv Coverage", fontweight="bold")
ax.set_ylabel("%")
ax.legend(fontsize=8, framealpha=0.9)

fig.suptitle("HuggingFace CJK+EN Model Analysis Overview", fontsize=15, fontweight="bold", y=1.01)
save(fig, "00_overview.png")

print("\nAll figures saved to:", OUT_DIR)
