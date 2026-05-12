#!/bin/bash
# CJK + EN 모델 메타데이터 전체 수집 스크립트
# Usage: cd scripts && bash run_all.sh
#
# 중단 후 재실행 시:
#   bash run_all.sh --skip-stage1             # Stage 1(ID수집) 건너뜀
#   bash run_all.sh --skip-stage1 --skip-expand  # Stage 1+1.5 모두 건너뜀
#
# BFS 재귀 깊이 조절:
#   bash run_all.sh --max-depth 1    # 직접 파생모델만 (기본값)
#   bash run_all.sh --max-depth 2    # 파생의 파생까지
#   bash run_all.sh --max-depth -1   # 무제한 재귀
#   bash run_all.sh --max-depth 0    # BFS 비활성화 (시드 ID만)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ARGS="$@"

echo "============================================"
echo " HuggingFace Model Metadata Collection"
echo " Languages: ko, ja, zh, en"
echo " Args: ${ARGS:-none}"
echo "============================================"

for lang in ko ja zh en; do
    echo ""
    echo "============================================"
    echo " Starting: ${lang}"
    echo " $(date)"
    echo "============================================"

    uv run python model_metadata_scraping.py --lang "$lang" $ARGS

    echo ""
    echo " Done: ${lang} ($(date))"
    echo "============================================"
done

echo ""
echo "============================================"
echo " All languages complete!"
echo " $(date)"
echo "============================================"

# Summary
echo ""
echo "Output files:"
for lang in ko ja zh en; do
    if [ -f "model_meta_${lang}.jsonl" ]; then
        count=$(wc -l < "model_meta_${lang}.jsonl")
        errors=$(grep -c '"_error": true' "model_meta_${lang}.jsonl" 2>/dev/null || echo 0)
        echo "  model_meta_${lang}.jsonl: ${count} records (${errors} errors)"
    fi
done
