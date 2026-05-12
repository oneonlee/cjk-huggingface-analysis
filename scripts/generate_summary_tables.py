"""
크롤링된 model_meta_{lang}.jsonl 데이터를 분석하여 빈도표 CSV를 생성한다.

Usage:
    cd scripts
    uv run python generate_summary_tables.py

Output (summary/ 디렉토리):
    01_language_overview.csv          — 언어별 전체 모델 수
    02_pipeline_tag_by_lang.csv       — 언어별 pipeline_tag(task) 분포
    03_library_by_lang.csv            — 언어별 프레임워크(library) 분포
    04_license_by_lang.csv            — 언어별 라이선스 분포
    05_org_vs_user_by_lang.csv        — 언어별 기관/개인 비율
    06_yearly_growth.csv              — 연도별 모델 등록 추이
    07_param_size_distribution.csv    — 파라미터 크기 구간별 분포
    08_top_authors_by_lang.csv        — 언어별 상위 저자(author) Top 20
    09_gated_private_disabled.csv     — 언어별 gated/private/disabled 비율
    10_model_tree_summary.csv         — 언어별 파생모델(tree) 통계
"""

import json
import os
import csv
from collections import Counter, defaultdict
from pathlib import Path

LANGS = ["ko", "ja", "zh", "en"]
SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "summary"
OUT_DIR.mkdir(exist_ok=True)


def load_all_data():
    """모든 언어의 JSONL 데이터를 로드한다."""
    data = {}  # lang -> list[dict]
    for lang in LANGS:
        path = SCRIPT_DIR / f"model_meta_{lang}.jsonl"
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("_error"):
                    continue
                records.append(rec)
        data[lang] = records
        print(f"  {lang}: {len(records):,} records loaded")
    return data


def write_csv(filename, header, rows):
    """CSV 파일을 작성한다."""
    path = OUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {filename} ({len(rows)} rows)")


# ── 01. 언어별 전체 모델 수 ──────────────────────────────────────
def table_language_overview(data):
    header = ["language", "model_count"]
    rows = [[lang, len(data[lang])] for lang in LANGS]
    rows.append(["total", sum(len(data[l]) for l in LANGS)])
    write_csv("01_language_overview.csv", header, rows)


# ── 02. pipeline_tag(task) 분포 ──────────────────────────────────
def table_pipeline_tag(data):
    # 전체 태그 수집
    all_tags = set()
    counters = {}
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            tag = rec.get("pipeline_tag") or "N/A"
            c[tag] += 1
            all_tags.add(tag)
        counters[lang] = c

    # 전체 합산 기준 내림차순 정렬
    total = Counter()
    for c in counters.values():
        total += c
    sorted_tags = sorted(all_tags, key=lambda t: total[t], reverse=True)

    header = ["pipeline_tag"] + LANGS + ["total"]
    rows = []
    for tag in sorted_tags:
        row = [tag] + [counters[l][tag] for l in LANGS] + [total[tag]]
        rows.append(row)
    write_csv("02_pipeline_tag_by_lang.csv", header, rows)


# ── 03. library 분포 ────────────────────────────────────────────
def table_library(data):
    all_libs = set()
    counters = {}
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            lib = rec.get("library_name") or "N/A"
            c[lib] += 1
            all_libs.add(lib)
        counters[lang] = c

    total = Counter()
    for c in counters.values():
        total += c
    sorted_libs = sorted(all_libs, key=lambda t: total[t], reverse=True)

    header = ["library_name"] + LANGS + ["total"]
    rows = []
    for lib in sorted_libs:
        row = [lib] + [counters[l][lib] for l in LANGS] + [total[lib]]
        rows.append(row)
    write_csv("03_library_by_lang.csv", header, rows)


# ── 04. 라이선스 분포 ───────────────────────────────────────────
def table_license(data):
    all_lics = set()
    counters = {}
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            lic = rec.get("license") or "N/A"
            if lic == "None":
                lic = "N/A"
            c[lic] += 1
            all_lics.add(lic)
        counters[lang] = c

    total = Counter()
    for c in counters.values():
        total += c
    sorted_lics = sorted(all_lics, key=lambda t: total[t], reverse=True)

    header = ["license"] + LANGS + ["total"]
    rows = []
    for lic in sorted_lics:
        row = [lic] + [counters[l][lic] for l in LANGS] + [total[lic]]
        rows.append(row)
    write_csv("04_license_by_lang.csv", header, rows)


# ── 05. 기관 vs 개인 비율 ───────────────────────────────────────
def table_org_vs_user(data):
    header = ["language", "organization", "individual", "org_ratio(%)"]
    rows = []
    for lang in LANGS:
        org = sum(1 for r in data[lang] if str(r.get("is_organization")) == "True")
        ind = len(data[lang]) - org
        ratio = round(org / len(data[lang]) * 100, 1) if data[lang] else 0
        rows.append([lang, org, ind, ratio])
    # total
    total_org = sum(int(r[1]) for r in rows)
    total_ind = sum(int(r[2]) for r in rows)
    total_all = total_org + total_ind
    rows.append(["total", total_org, total_ind, round(total_org / total_all * 100, 1) if total_all else 0])
    write_csv("05_org_vs_user_by_lang.csv", header, rows)


# ── 06. 연도별 모델 등록 추이 ───────────────────────────────────
def table_yearly_growth(data):
    counters = {}
    all_years = set()
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            created = rec.get("created_at") or ""
            if len(created) >= 4:
                year = created[:4]
                if year.isdigit():
                    c[year] += 1
                    all_years.add(year)
        counters[lang] = c

    sorted_years = sorted(all_years)
    header = ["year"] + LANGS + ["total"]
    rows = []
    for year in sorted_years:
        row = [year] + [counters[l][year] for l in LANGS] + [sum(counters[l][year] for l in LANGS)]
        rows.append(row)
    write_csv("06_yearly_growth.csv", header, rows)


# ── 07. 파라미터 크기 구간별 분포 ───────────────────────────────
def _param_bucket(param_count):
    """파라미터 수를 구간으로 분류한다."""
    if param_count is None:
        return "unknown"
    try:
        p = int(param_count)
    except (ValueError, TypeError):
        return "unknown"
    if p == 0:
        return "unknown"
    if p < 100_000_000:           # < 100M
        return "< 100M"
    elif p < 1_000_000_000:       # 100M – 1B
        return "100M–1B"
    elif p < 3_000_000_000:       # 1B – 3B
        return "1B–3B"
    elif p < 7_000_000_000:       # 3B – 7B
        return "3B–7B"
    elif p < 13_000_000_000:      # 7B – 13B
        return "7B–13B"
    elif p < 70_000_000_000:      # 13B – 70B
        return "13B–70B"
    else:                          # 70B+
        return "70B+"


BUCKET_ORDER = ["< 100M", "100M–1B", "1B–3B", "3B–7B", "7B–13B", "13B–70B", "70B+", "unknown"]


def table_param_distribution(data):
    counters = {}
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            bucket = _param_bucket(rec.get("param_count"))
            c[bucket] += 1
        counters[lang] = c

    header = ["param_size"] + LANGS + ["total"]
    rows = []
    for bucket in BUCKET_ORDER:
        row = [bucket] + [counters[l][bucket] for l in LANGS] + [sum(counters[l][bucket] for l in LANGS)]
        rows.append(row)
    write_csv("07_param_size_distribution.csv", header, rows)


# ── 08. 언어별 상위 저자 Top 20 ─────────────────────────────────
def table_top_authors(data):
    header = ["rank", "language", "author", "model_count", "is_organization"]
    rows = []
    for lang in LANGS:
        author_counts = Counter()
        author_is_org = {}
        for rec in data[lang]:
            author = rec.get("author") or "unknown"
            author_counts[author] += 1
            author_is_org[author] = str(rec.get("is_organization")) == "True"
        for rank, (author, cnt) in enumerate(author_counts.most_common(20), 1):
            rows.append([rank, lang, author, cnt, author_is_org.get(author, False)])
    write_csv("08_top_authors_by_lang.csv", header, rows)


# ── 09. gated / private / disabled 비율 ─────────────────────────
def table_gated_private(data):
    header = ["language", "total", "gated", "gated(%)", "private", "private(%)", "disabled", "disabled(%)"]
    rows = []
    for lang in LANGS:
        n = len(data[lang])
        gated = sum(1 for r in data[lang] if str(r.get("gated")) not in ("False", "None", ""))
        private = sum(1 for r in data[lang] if str(r.get("private")) == "True")
        disabled = sum(1 for r in data[lang] if str(r.get("disabled")) == "True")
        rows.append([
            lang, n,
            gated, round(gated / n * 100, 1) if n else 0,
            private, round(private / n * 100, 1) if n else 0,
            disabled, round(disabled / n * 100, 1) if n else 0,
        ])
    write_csv("09_gated_private_disabled.csv", header, rows)


# ── 10. 파생모델(tree) 통계 ─────────────────────────────────────
def table_model_tree(data):
    header = [
        "language",
        "adapters_total", "adapters_mean", "adapters_max",
        "finetunes_total", "finetunes_mean", "finetunes_max",
        "quantizations_total", "quantizations_mean", "quantizations_max",
        "merges_total", "merges_mean", "merges_max",
    ]
    rows = []
    for lang in LANGS:
        n = len(data[lang])
        stats = {}
        for key in ["tree_adapters", "tree_finetunes", "tree_quantizations", "tree_merges"]:
            vals = [r.get(key, 0) or 0 for r in data[lang]]
            vals = [int(v) for v in vals]
            total = sum(vals)
            mean = round(total / n, 2) if n else 0
            mx = max(vals) if vals else 0
            stats[key] = (total, mean, mx)
        rows.append([
            lang,
            *stats["tree_adapters"],
            *stats["tree_finetunes"],
            *stats["tree_quantizations"],
            *stats["tree_merges"],
        ])
    write_csv("10_model_tree_summary.csv", header, rows)


# ── 11. 월별 모델 등록 추이 ─────────────────────────────────────
def table_monthly_growth(data):
    counters = {}
    all_months = set()
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            created = rec.get("created_at") or ""
            if len(created) >= 7:
                month = created[:7]  # "YYYY-MM"
                c[month] += 1
                all_months.add(month)
        counters[lang] = c

    sorted_months = sorted(all_months)
    header = ["year_month"] + LANGS + ["total"]
    rows = []
    for month in sorted_months:
        row = [month] + [counters[l][month] for l in LANGS] + [sum(counters[l][month] for l in LANGS)]
        rows.append(row)
    write_csv("11_monthly_growth.csv", header, rows)


# ── 12. param_source 분포 ───────────────────────────────────────
def table_param_source(data):
    all_sources = set()
    counters = {}
    for lang in LANGS:
        c = Counter()
        for rec in data[lang]:
            src = rec.get("param_source") or "N/A"
            if src == "None":
                src = "N/A"
            c[src] += 1
            all_sources.add(src)
        counters[lang] = c

    total = Counter()
    for c in counters.values():
        total += c
    sorted_srcs = sorted(all_sources, key=lambda t: total[t], reverse=True)

    header = ["param_source"] + LANGS + ["total"]
    rows = []
    for src in sorted_srcs:
        row = [src] + [counters[l][src] for l in LANGS] + [total[src]]
        rows.append(row)
    write_csv("12_param_source_distribution.csv", header, rows)


# ── 13. arxiv 논문 보유 비율 ────────────────────────────────────
def table_arxiv(data):
    header = ["language", "total", "has_arxiv", "has_arxiv(%)", "no_arxiv", "no_arxiv(%)"]
    rows = []
    for lang in LANGS:
        n = len(data[lang])
        has = sum(1 for r in data[lang] if r.get("arxiv_id") and str(r.get("arxiv_id")) != "None")
        no = n - has
        rows.append([
            lang, n,
            has, round(has / n * 100, 1) if n else 0,
            no, round(no / n * 100, 1) if n else 0,
        ])
    write_csv("13_arxiv_coverage.csv", header, rows)


def main():
    print("Loading data...")
    data = load_all_data()
    print()
    print("Generating summary tables...")

    table_language_overview(data)
    table_pipeline_tag(data)
    table_library(data)
    table_license(data)
    table_org_vs_user(data)
    table_yearly_growth(data)
    table_param_distribution(data)
    table_top_authors(data)
    table_gated_private(data)
    table_model_tree(data)
    table_monthly_growth(data)
    table_param_source(data)
    table_arxiv(data)

    print()
    print(f"Done! All CSV files saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
