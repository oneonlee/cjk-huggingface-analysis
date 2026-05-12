# CJK HuggingFace Analysis

HuggingFace 데이터셋 및 모델의 메타데이터를 수집·분석하는 프로젝트.

## 프로젝트 구조

```
scripts/
  hugging_metadata_scraping.py   # 데이터셋 메타데이터 크롤링
  hugging_card_scraping.py       # 데이터셋 카드(README) 크롤링
  model_metadata_scraping.py     # 모델 메타데이터 크롤링
  model_card_scraping.py         # 모델 카드(README) 크롤링
analysis/
  analysis_metadata.ipynb        # 데이터셋 메타데이터 분석
  analysis_datasetcard.ipynb     # 데이터셋 카드 분석
```

## 실행 방법

### 모델 메타데이터 수집

```bash
cd scripts

# API_TOKEN 설정 필요 (model_metadata_scraping.py 내 API_TOKEN 변수)
python model_metadata_scraping.py --lang ko --max-pages 50
python model_metadata_scraping.py --lang ja --max-pages 50
python model_metadata_scraping.py --lang zh --max-pages 50
python model_metadata_scraping.py --lang en --max-pages 50
```

출력: `model_meta_{lang}.csv`

### 모델 카드 수집

```bash
# username, token 설정 필요 (model_card_scraping.py 내 변수)
python model_card_scraping.py --lang ko
```

입력: `./data/model_meta/model_meta_{lang}.csv`
출력: `./data/model_card/model_cards_{lang}.csv`

### 데이터셋 메타데이터 수집 (기존)

```bash
# lang 변수와 API_TOKEN을 코드 내에서 직접 수정
python hugging_metadata_scraping.py
```

## 주요 설정

- **API_TOKEN**: HuggingFace API 토큰 (Bearer 인증). 각 스크립트 상단에 설정.
- **username / token**: 모델/데이터셋 카드 수집 시 git clone 인증에 사용.
- **--max-pages**: 모델 목록 페이지 최대 크롤링 수 (기본 50).
- **Rate limiting**: 페이지 간 1초, API 호출 간 0.5초, 기관 확인 0.2초 딜레이.

## 모델 메타데이터 수집 필드

| 필드 | 설명 |
|------|------|
| `id` | 모델 경로 (author/model_name) |
| `author` | 모델 저자/기관 |
| `is_organization` | 기관 여부 |
| `org_name` | 기관명 (개인이면 None) |
| `downloads_30` | 최근 30일 다운로드 수 |
| `param_count` | 파라미터 수 |
| `param_source` | 파라미터 수 출처 (safetensors/config_json/tag/unknown) |
| `tree_adapters` / `tree_finetunes` / `tree_quantizations` / `tree_merges` | 파생모델 수 |
| `tags` | 메타데이터 태그 |
| `license` | 라이선스 |
| `pipeline_tag` | 모델 태스크 유형 |
| `library_name` | 사용 라이브러리 |

## 의존성

```
requests
beautifulsoup4
pandas
pyyaml
gitpython
tqdm
```

## 데이터 수집 전략

- **Stage 1**: HTML 페이지 파싱으로 모델 ID 목록 수집
- **Stage 2**: REST API + HTML 스크래핑으로 상세 메타데이터 수집
- **파라미터 수**: safetensors API → config.json → tags 순으로 fallback
- **기관 확인**: `/api/organizations/{author}/members` 호출 + 딕셔너리 캐싱
- **Model tree**: HTML 페이지에서 파생모델 카운트 파싱
