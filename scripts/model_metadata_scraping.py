import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import json
import datetime
import argparse
import re
import os

# ============================================================
# Configuration
# ============================================================

# HuggingFace API tokens (2개 준비, rate-limit 시 자동 교체)
# 환경변수 HF_TOKEN_1, HF_TOKEN_2 로 설정하거나 아래에 직접 입력
import os
API_TOKENS = [
    os.environ.get("HF_TOKEN_1", ""),  # Token 1
    os.environ.get("HF_TOKEN_2", ""),  # Token 2
]

API_URL = "https://huggingface.co/api/models"

# 언어별 태그 매핑 (HuggingFace가 인식하는 태그)
# 같은 언어에 대해 여러 태그를 크롤링한 뒤 dedup
LANGUAGE_TAGS = {
    'ko': ['ko', 'kor'],
    'ja': ['ja', 'jpn', 'jap'],
    'zh': ['zh', 'zho'],
    'en': ['en', 'eng'],
}

# 정렬 기준 (downloads, likes 두 가지로 크롤링하여 커버리지 확보)
SORT_ORDERS = ['downloads', 'likes']

# API 페이지당 모델 수
PAGE_SIZE = 100

# 최대 페이지 수 (기본값 100, 각 tag+sort 조합당)
MAX_PAGES = 100

# Rate limit 관련 설정
RATE_LIMIT_WAIT_BASE = 60  # 두 토큰 모두 rate-limit 시 대기 초


# ============================================================
# Token rotation with rate-limit handling
# ============================================================

class TokenRotator:
    """API 토큰 2개를 관리하며, rate-limit 시 자동 교체 및 대기."""

    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t]  # 빈 문자열 제거
        self.current_idx = 0
        # 각 토큰별 rate-limit 해제 예상 시각
        self.blocked_until = [0.0] * len(self.tokens) if self.tokens else [0.0]

    def _get_headers(self, idx):
        if not self.tokens:
            return {}
        return {"Authorization": f"Bearer {self.tokens[idx]}"}

    def get_headers(self):
        """현재 사용 가능한 토큰의 headers를 반환. 둘 다 blocked이면 sleep 후 반환."""
        if not self.tokens:
            return {}

        now = time.time()

        # 현재 토큰이 사용 가능하면 바로 반환
        if self.blocked_until[self.current_idx] <= now:
            return self._get_headers(self.current_idx)

        # 다른 토큰 시도
        other_idx = (self.current_idx + 1) % len(self.tokens)
        if len(self.tokens) > 1 and self.blocked_until[other_idx] <= now:
            self.current_idx = other_idx
            print(f"  [TokenRotator] Switched to token #{self.current_idx + 1}")
            return self._get_headers(self.current_idx)

        # 둘 다 blocked — 가장 빨리 풀리는 시각까지 대기
        earliest = min(self.blocked_until)
        wait = max(earliest - now, 1)
        print(f"  [TokenRotator] All tokens rate-limited. Sleeping {wait:.0f}s ...")
        time.sleep(wait)
        # 대기 후 해제된 토큰 선택
        now = time.time()
        for i in range(len(self.tokens)):
            if self.blocked_until[i] <= now:
                self.current_idx = i
                break
        return self._get_headers(self.current_idx)

    def mark_rate_limited(self, retry_after=None):
        """현재 토큰을 rate-limited로 마킹하고 다른 토큰으로 교체 시도."""
        wait = retry_after if retry_after else RATE_LIMIT_WAIT_BASE
        self.blocked_until[self.current_idx] = time.time() + wait
        print(f"  [TokenRotator] Token #{self.current_idx + 1} rate-limited for {wait}s")

        if len(self.tokens) > 1:
            other_idx = (self.current_idx + 1) % len(self.tokens)
            if self.blocked_until[other_idx] <= time.time():
                self.current_idx = other_idx
                print(f"  [TokenRotator] Switched to token #{self.current_idx + 1}")


# ============================================================
# Core functions
# ============================================================

def api_request(url, token_rotator, max_retries=3, timeout=15):
    """Rate-limit aware API 요청. 429 시 토큰 교체 후 재시도."""
    for attempt in range(max_retries):
        headers = token_rotator.get_headers()
        try:
            response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_WAIT_BASE))
                token_rotator.mark_rate_limited(retry_after)
                continue  # 다음 attempt에서 교체된 토큰 사용

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1 * (attempt + 1))

    return None


def parse_link_header(response):
    """Link 헤더에서 next cursor URL 추출."""
    link_header = response.headers.get('Link', '')
    if not link_header:
        return None
    # Format: <URL>; rel="next"
    for part in link_header.split(','):
        part = part.strip()
        if 'rel="next"' in part:
            url = part.split(';')[0].strip().strip('<>')
            return url
    return None


def list_models_api(lang_tag, sort_order, token_rotator, max_pages=100):
    """API cursor pagination으로 모델 ID 목록 수집."""
    url = f"{API_URL}?language={lang_tag}&sort={sort_order}&limit={PAGE_SIZE}"
    model_ids = []

    for page in range(max_pages):
        resp = api_request(url, token_rotator, max_retries=3)
        if resp is None:
            break

        models = resp.json()
        if not models:
            break

        for m in models:
            mid = m.get('id') or m.get('modelId')
            if mid:
                model_ids.append(mid)

        # Follow cursor pagination via Link header
        next_url = parse_link_header(resp)
        if not next_url:
            break
        url = next_url

        time.sleep(0.3)

    return model_ids


def get_param_count(model_id, api_info, token_rotator):
    # Strategy 1: safetensors metadata from API
    safetensors = api_info.get('safetensors', {})
    if safetensors:
        params = safetensors.get('parameters', {})
        if params:
            if isinstance(params, dict):
                total = sum(params.values())
            else:
                total = params
            return total, 'safetensors'
        total = safetensors.get('total', None)
        if total:
            return total, 'safetensors'

    # Strategy 2: fetch config.json from repo
    try:
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        resp = api_request(config_url, token_rotator, max_retries=1, timeout=10)
        if resp and resp.status_code == 200:
            config = resp.json()
            if 'num_parameters' in config:
                return config['num_parameters'], 'config_json'
    except Exception:
        pass

    # Strategy 3: check tags for params
    for tag in api_info.get('tags', []):
        if tag.startswith('params:'):
            return tag.split(':')[1], 'tag'

    return None, 'unknown'


def get_derivative_ids(model_id, token_rotator, max_pages=50):
    """주어진 모델의 파생모델 ID 목록을 API로 수집 (BFS Stage 1.5용).
    HuggingFace API: /api/models?base_model_id={model_id} 로 파생모델 조회.
    max_pages * PAGE_SIZE 개까지 수집 (기본 50 * 100 = 5000개).
    """
    derivative_ids = []
    url = f"{API_URL}?filter=base_model:{model_id}&limit={PAGE_SIZE}"

    for _ in range(max_pages):
        resp = api_request(url, token_rotator, max_retries=2, timeout=15)
        if resp is None:
            break
        try:
            models = resp.json()
        except Exception:
            break
        if not models:
            break
        for m in models:
            mid = m.get('id') or m.get('modelId')
            if mid:
                derivative_ids.append(mid)
        next_url = parse_link_header(resp)
        if not next_url:
            break
        url = next_url
        time.sleep(0.2)

    return derivative_ids


def extract_model_tree(soup):
    tree = {
        'tree_adapters': 0,
        'tree_finetunes': 0,
        'tree_quantizations': 0,
        'tree_merges': 0,
    }

    try:
        tree_links = soup.find_all('a', href=re.compile(r'base_model'))
        for link in tree_links:
            text = link.get_text(strip=True).lower()
            count_match = re.search(r'(\d+)', text)
            if not count_match:
                parent_text = link.parent.get_text(strip=True) if link.parent else ''
                count_match = re.search(r'(\d+)', parent_text)

            count = int(count_match.group(1)) if count_match else 0
            href = link.get('href', '')

            if 'adapter' in href.lower() or 'adapter' in text:
                tree['tree_adapters'] = count
            elif 'finetune' in href.lower() or 'finetune' in text:
                tree['tree_finetunes'] = count
            elif 'quantized' in href.lower() or 'quantiz' in text:
                tree['tree_quantizations'] = count
            elif 'merge' in href.lower() or 'merge' in text:
                tree['tree_merges'] = count
    except Exception as e:
        print(f"Error extracting model tree: {e}")

    return tree


# Cache for organization checks
org_cache = {}


def check_is_organization(author, token_rotator):
    if author in org_cache:
        return org_cache[author]

    result = {'is_org': False, 'org_name': None}

    try:
        resp = api_request(
            f"https://huggingface.co/api/organizations/{author}/members",
            token_rotator, max_retries=2, timeout=10
        )
        if resp and resp.status_code == 200:
            result['is_org'] = True
            try:
                org_page = requests.get(
                    f"https://huggingface.co/{author}",
                    timeout=15
                )
                if org_page.status_code == 200:
                    org_soup = BeautifulSoup(org_page.text, 'html.parser')
                    title_tag = org_soup.find('title')
                    if title_tag:
                        title_text = title_tag.get_text(strip=True)
                        org_name = title_text.split(' - ')[0].strip()
                        if org_name:
                            result['org_name'] = org_name
            except Exception:
                result['org_name'] = author
        time.sleep(0.2)
    except Exception:
        pass

    org_cache[author] = result
    return result


def get_model_info(model_id, token_rotator):
    api_url = f"{API_URL}/{model_id}"

    response = api_request(api_url, token_rotator, max_retries=3)
    if response is None:
        return None

    api_info = response.json()

    # Get parameter count
    param_count, param_source = get_param_count(model_id, api_info, token_rotator)
    api_info['param_count'] = param_count
    api_info['param_source'] = param_source

    # HTML scraping for supplementary data (model tree, arxiv)
    web_url = f"https://huggingface.co/{model_id}"
    try:
        response = requests.get(web_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        arxiv_link = soup.find('a', href=lambda href: href and 'arxiv.org/abs/' in href)
        api_info['arxiv_id'] = arxiv_link['href'].split('/')[-1] if arxiv_link else None

        model_tree = extract_model_tree(soup)
        api_info.update(model_tree)

        if 'cardData' not in api_info or not api_info.get('cardData'):
            try:
                header_div = soup.find('div', {'data-target': 'ModelHeader'})
                if header_div:
                    header_props = json.loads(header_div['data-props'])
                    model_data = header_props.get('model', {})
                    card_data = model_data.get('cardData', {})
                    api_info['cardData'] = card_data
            except Exception:
                pass

    except Exception as e:
        print(f"Error scraping webpage for {model_id}: {e}")
        api_info['arxiv_id'] = None
        api_info.update({'tree_adapters': 0, 'tree_finetunes': 0,
                         'tree_quantizations': 0, 'tree_merges': 0})

    return api_info


def process_model_info(info, token_rotator):
    if not info:
        return None

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        return ' '.join(text.replace('\n', ' ').replace('\t', ' ').replace(';', ' ').split())

    created_at = None
    if '_id' in info:
        try:
            timestamp_hex = info['_id'][:8]
            timestamp = int(timestamp_hex, 16)
            created_at = datetime.datetime.fromtimestamp(timestamp).isoformat()
        except Exception:
            pass

    card_data = info.get('cardData', {}) or {}
    languages = card_data.get('language', [])
    if isinstance(languages, str):
        languages = [languages]

    license_info = card_data.get('license', info.get('license', None))
    if isinstance(license_info, list):
        license_value = license_info[0] if license_info else None
    elif isinstance(license_info, str):
        license_value = license_info
    else:
        license_value = None

    if not license_value:
        for tag in info.get('tags', []):
            if tag.startswith('license:'):
                license_value = tag.split(':', 1)[1]
                break

    author = info.get('author', '')
    org_info = check_is_organization(author, token_rotator)

    return {
        'id': clean_text(info.get('id', 'None')),
        'author': clean_text(info.get('author', 'None')),
        'is_organization': org_info['is_org'],
        'org_name': clean_text(org_info['org_name']) if org_info['org_name'] else 'None',
        'created_at': created_at or 'None',
        'lastModified': clean_text(info.get('lastModified', 'None')),
        'downloads_30': int(info.get('downloads', 0) or 0),
        'likes': int(info.get('likes', 0) or 0),
        'tags': ', '.join(info.get('tags', [])) or 'None',
        'pipeline_tag': clean_text(info.get('pipeline_tag', 'None')),
        'library_name': clean_text(info.get('library_name', 'None')),
        'languages': ', '.join(languages) or 'None',
        'license': clean_text(license_value) if license_value else 'None',
        'param_count': str(info.get('param_count', 'None')) if info.get('param_count') else 'None',
        'param_source': info.get('param_source', 'unknown'),
        'tree_adapters': int(info.get('tree_adapters', 0)),
        'tree_finetunes': int(info.get('tree_finetunes', 0)),
        'tree_quantizations': int(info.get('tree_quantizations', 0)),
        'tree_merges': int(info.get('tree_merges', 0)),
        'private': str(info.get('private', False)),
        'gated': str(info.get('gated', False)),
        'disabled': str(info.get('disabled', False)),
        'arxiv_id': clean_text(info.get('arxiv_id', 'None')) if info.get('arxiv_id') else 'None',
        'description': clean_text(info.get('description', ''))[:200] or 'None',
    }


def main():
    parser = argparse.ArgumentParser(description='Collect HuggingFace model metadata')
    parser.add_argument('--lang', type=str, required=True,
                        choices=list(LANGUAGE_TAGS.keys()),
                        help='Language group (ko, ja, zh, en)')
    parser.add_argument('--max-pages', type=int, default=MAX_PAGES,
                        help=f'Max pages per tag+sort combination (default: {MAX_PAGES})')
    parser.add_argument('--skip-stage1', action='store_true',
                        help='Skip Stage 1 (ID collection), reuse saved ID list')
    parser.add_argument('--skip-expand', action='store_true',
                        help='Skip Stage 1.5 (BFS derivative expansion)')
    parser.add_argument('--max-depth', type=int, default=1,
                        help='BFS max depth for derivative expansion. '
                             '1=직접 파생만, -1=무제한 재귀 (기본: 1)')
    args = parser.parse_args()

    lang_tags = LANGUAGE_TAGS[args.lang]
    token_rotator = TokenRotator(API_TOKENS)

    id_list_path = f'model_ids_{args.lang}.json'
    expanded_path = f'model_ids_{args.lang}_expanded.json'
    explored_path = f'model_ids_{args.lang}_explored.json'
    bfs_state_path = f'bfs_state_{args.lang}.json'
    frontier_path = f'bfs_frontier_{args.lang}.json'  # depth 시작 전 저장 (중단 복구용)
    depth_map_path = f'bfs_depth_{args.lang}.json'    # {model_id: depth} — 나중에 depth 기준 필터링용
    output_path = f'model_meta_{args.lang}.jsonl'

    # Stage 1: Collect model IDs via API
    if args.skip_stage1 and os.path.exists(id_list_path):
        with open(id_list_path, 'r', encoding='utf-8') as f:
            all_model_ids = json.load(f)
        print(f"Stage 1 skipped. Loaded {len(all_model_ids)} model IDs from {id_list_path}")
    else:
        all_model_ids = []
        seen = set()

        for lang_tag in lang_tags:
            for sort_order in SORT_ORDERS:
                print(f"\n=== Listing models: tag={lang_tag}, sort={sort_order} "
                      f"(max {args.max_pages} pages x {PAGE_SIZE}/page) ===")

                model_ids = list_models_api(
                    lang_tag, sort_order, token_rotator, max_pages=args.max_pages
                )

                combo_count = 0
                for mid in model_ids:
                    if mid not in seen:
                        seen.add(mid)
                        all_model_ids.append(mid)
                        combo_count += 1

                print(f"  [{lang_tag}/{sort_order}]: {len(model_ids)} fetched, "
                      f"{combo_count} new unique")

        # Save ID list for resume
        with open(id_list_path, 'w', encoding='utf-8') as f:
            json.dump(all_model_ids, f, ensure_ascii=False)
        print(f"\nSaved {len(all_model_ids)} model IDs to {id_list_path}")

    print(f"Total unique models (seed): {len(all_model_ids)}")

    # depth_map 초기화: 시드 모델은 depth=0
    depth_map = {}
    if os.path.exists(depth_map_path):
        with open(depth_map_path, 'r', encoding='utf-8') as f:
            depth_map = json.load(f)
    for mid in all_model_ids:
        if mid not in depth_map:
            depth_map[mid] = 0

    # ============================================================
    # Stage 1.5: BFS derivative expansion
    # ============================================================
    if args.skip_expand:
        # --skip-expand: load expanded list if exists, else use seed
        if os.path.exists(expanded_path):
            with open(expanded_path, 'r', encoding='utf-8') as f:
                all_model_ids = json.load(f)
            print(f"Stage 1.5 skipped. Loaded {len(all_model_ids)} expanded IDs from {expanded_path}")
        else:
            print("Stage 1.5 skipped (no expanded file found, using seed IDs).")
    elif args.max_depth == 0:
        print("Stage 1.5 skipped (--max-depth 0).")
    else:
        # Load BFS state: 이전 실행에서 완료한 depth 수
        bfs_state = {'depth_completed': 0}
        if os.path.exists(bfs_state_path):
            with open(bfs_state_path, 'r', encoding='utf-8') as f:
                bfs_state = json.load(f)
        depth_completed = bfs_state['depth_completed']

        if args.max_depth < 0:
            remaining_depth = -1  # 무제한
        else:
            remaining_depth = args.max_depth - depth_completed

        depth_label = '∞' if args.max_depth < 0 else str(args.max_depth)
        print(f"\n=== Stage 1.5: BFS derivative expansion "
              f"(target={depth_label}, done={depth_completed}, "
              f"remaining={remaining_depth if remaining_depth >= 0 else '∞'}) ===")

        if remaining_depth == 0:
            print(f"Already completed depth {depth_completed}. Skipping.")
            if os.path.exists(expanded_path):
                with open(expanded_path, 'r', encoding='utf-8') as f:
                    all_model_ids = json.load(f)
                print(f"Loaded {len(all_model_ids)} IDs from {expanded_path}")
        else:
            # Load explored set
            explored = set()
            if os.path.exists(explored_path):
                with open(explored_path, 'r', encoding='utf-8') as f:
                    explored = set(json.load(f))
                print(f"Resuming BFS: {len(explored)} models already explored.")

            # Load expanded list
            if os.path.exists(expanded_path):
                with open(expanded_path, 'r', encoding='utf-8') as f:
                    all_model_ids = json.load(f)
                print(f"Resuming BFS: loaded {len(all_model_ids)} IDs from {expanded_path}")

            seen = set(all_model_ids)

            def _run_one_depth(level, abs_depth):
                """한 depth 레벨을 처리하고 next_level 반환. expanded/explored/bfs_state/depth_map 저장."""
                next_level = []
                print(f"\n[BFS Depth {abs_depth}] Expanding {len(level)} models ...")

                # depth 시작 전 frontier 저장 (중단 시 이 레벨을 정확히 재개하기 위해)
                with open(frontier_path, 'w', encoding='utf-8') as f:
                    json.dump(level, f, ensure_ascii=False)

                for model_id in tqdm(level, desc=f"BFS Depth {abs_depth}"):
                    if model_id in explored:
                        continue  # 이미 탐색됨 (중단 후 재개 시 건너뜀)
                    deriv_ids = get_derivative_ids(model_id, token_rotator)
                    for did in deriv_ids:
                        if did not in seen:
                            seen.add(did)
                            next_level.append(did)
                            all_model_ids.append(did)
                            depth_map[did] = abs_depth  # 발견된 depth 기록
                    explored.add(model_id)
                    time.sleep(0.1)

                # depth 완료 후 저장
                with open(expanded_path, 'w', encoding='utf-8') as f:
                    json.dump(all_model_ids, f, ensure_ascii=False)
                with open(explored_path, 'w', encoding='utf-8') as f:
                    json.dump(list(explored), f, ensure_ascii=False)
                with open(depth_map_path, 'w', encoding='utf-8') as f:
                    json.dump(depth_map, f, ensure_ascii=False)
                bfs_state['depth_completed'] = abs_depth
                with open(bfs_state_path, 'w', encoding='utf-8') as f:
                    json.dump(bfs_state, f)
                if os.path.exists(frontier_path):
                    os.remove(frontier_path)  # depth 완료 → frontier 제거

                print(f"[BFS Depth {abs_depth}] New: {len(next_level)}, Total: {len(all_model_ids)}")
                return next_level

            # ── 중단 복구: frontier 파일이 있으면 그 depth가 미완료 상태 ──
            if os.path.exists(frontier_path):
                with open(frontier_path, 'r', encoding='utf-8') as f:
                    interrupted_level = json.load(f)
                abs_depth = depth_completed + 1
                remaining_in_level = [m for m in interrupted_level if m not in explored]
                print(f"Resuming interrupted depth {abs_depth} "
                      f"({len(remaining_in_level)}/{len(interrupted_level)} models left)...")
                next_level = _run_one_depth(remaining_in_level, abs_depth)
                depth_completed = abs_depth
                # 남은 remaining 재계산
                if args.max_depth >= 0:
                    remaining_depth = args.max_depth - depth_completed
                current_level = next_level
            else:
                # 깔끔한 시작 또는 이전 depth가 정상 완료된 경우
                current_level = [mid for mid in all_model_ids if mid not in explored]

            # ── 남은 depth 반복 ──
            iters = 0
            while current_level and (remaining_depth < 0 or iters < remaining_depth):
                iters += 1
                abs_depth = depth_completed + iters
                current_level = _run_one_depth(current_level, abs_depth)

            print(f"\nBFS expansion complete. depth_completed={bfs_state['depth_completed']}, "
                  f"Total model IDs: {len(all_model_ids)}")

    print(f"Total unique models (after expansion): {len(all_model_ids)}")

    # Stage 2: Fetch details with incremental saving
    # Resume: load already-fetched IDs from existing JSONL
    # 동시에 bfs_depth 필드가 없는 기존 레코드를 depth_map 기준으로 채워준다 (마이그레이션)
    done_ids = set()
    needs_migration = []  # (line_index, id) — bfs_depth 없는 기존 레코드
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        mid = record.get('id', '')
                        done_ids.add(mid)
                        if not record.get('_error') and 'bfs_depth' not in record:
                            needs_migration.append((i, mid))
                    except json.JSONDecodeError:
                        continue
        print(f"Resuming: {len(done_ids)} models already collected, skipping.")

        # bfs_depth 없는 기존 레코드 마이그레이션: depth_map에서 가져오거나 0으로 설정
        if needs_migration:
            print(f"Migrating {len(needs_migration)} records without bfs_depth field ...")
            lines = open(output_path, 'r', encoding='utf-8').readlines()
            migrated = 0
            for idx, mid in needs_migration:
                try:
                    record = json.loads(lines[idx])
                    if 'bfs_depth' not in record and not record.get('_error'):
                        record['bfs_depth'] = depth_map.get(mid, 0)
                        lines[idx] = json.dumps(record, ensure_ascii=False) + '\n'
                        migrated += 1
                except Exception:
                    continue
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"Migration complete: {migrated} records updated.")

    remaining = [mid for mid in all_model_ids if mid not in done_ids]
    print(f"Models to fetch: {len(remaining)}")

    collected = len(done_ids)
    with open(output_path, 'a', encoding='utf-8') as f:
        for model_id in tqdm(remaining, desc="Fetching model details"):
            try:
                info = get_model_info(model_id, token_rotator)
                processed = process_model_info(info, token_rotator)
                if processed:
                    processed['bfs_depth'] = depth_map.get(model_id, 0)
                    f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                    f.flush()
                    collected += 1
                else:
                    # API 실패 (gated/approve 필요 등) — 에러 기록 저장
                    error_record = {
                        'id': model_id,
                        '_error': True,
                        '_error_reason': 'api_failed_or_gated',
                    }
                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    f.flush()
                    print(f"  [SKIP] {model_id}: saved error record (gated/approve?)")
            except Exception as e:
                # 예외 발생 시에도 에러 기록 저장 → 재실행 시 skip
                error_record = {
                    'id': model_id,
                    '_error': True,
                    '_error_reason': str(e)[:200],
                }
                try:
                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    f.flush()
                except Exception:
                    pass
                print(f"  [ERROR] {model_id}: {e}")
            time.sleep(0.5)

    # Stage 3: Summary
    print(f"\nResults saved to: {output_path}")

    records = []
    error_count = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get('_error'):
                        error_count += 1
                    else:
                        records.append(r)
                except json.JSONDecodeError:
                    continue

    print(f"Total models collected: {len(records)}")
    print(f"Error/skipped records: {error_count}")

    if records:
        records.sort(key=lambda x: x.get('downloads_30', 0), reverse=True)
        print("\nTop 10 models by downloads:")
        for r in records[:10]:
            print(f"  {r['id']:50s}  dl={r['downloads_30']}  "
                  f"likes={r['likes']}  params={r['param_count']}  "
                  f"org={r['is_organization']}")

        org_count = sum(1 for r in records if r.get('is_organization'))
        print(f"\nModels from organizations: {org_count}/{len(records)}")


if __name__ == "__main__":
    main()
