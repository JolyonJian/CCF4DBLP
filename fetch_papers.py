import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime
import logging
import os
import time
import random
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------
# 配置日志
# ---------------------
LOG_FILE = "./output/fetch_papers.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    ]
)

# ---------------------
# Session/Retry & 缓存/节流 实现
# ---------------------
def create_session(retries=3, backoff_factor=0.5):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; PaperFetcher/1.0; +https://your.email.or.site/)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    })
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor,
        raise_on_status=False,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# 文件缓存，用于减少重复请求
def _cache_get_path(cache_dir, url):
    h = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{h}.html")

# 全局节流时间（记录上一次请求时间）
_last_request_time = 0.0

def fetch_with_backoff(session, url, min_interval=1.5, max_retries=8, timeout=15, cache_dir='cache', use_cache=True):
    """
    使用 session 安全请求：
    - 最小请求间隔 min_interval（秒）
    - 遇到 429 尊重 Retry-After 或指数退避（带随机抖动）
    - 可选本地缓存（缓存成功的 200 响应）
    返回 requests.Response 或 None（失败）
    """
    global _last_request_time

    # 尝试从缓存读取（如果开启）
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _cache_get_path(cache_dir, url)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                r = requests.Response()
                r.status_code = 200
                r._content = text.encode('utf-8')
                r.headers = {}
                logging.info(f"Loaded from cache: {url}")
                return r
            except Exception as e:
                logging.warning(f"Failed to read cache for {url}: {e}")

    attempt = 0
    while attempt < max_retries:
        # 节流：确保与上次请求间隔至少 min_interval
        now = time.time()
        wait = min_interval - (now - _last_request_time)
        if wait > 0:
            sleep_time = wait + random.random() * 0.5
            time.sleep(sleep_time)

        try:
            resp = session.get(url, timeout=timeout)
            _last_request_time = time.time()
        except requests.RequestException as e:
            logging.warning(f"Request error for {url}: {e}. attempt={attempt}")
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)
            attempt += 1
            continue

        status = resp.status_code
        if status == 200:
            # 缓存到磁盘（异步更好，这里同步写入）
            if use_cache and cache_dir:
                try:
                    with open(_cache_get_path(cache_dir, url), 'w', encoding='utf-8') as f:
                        f.write(resp.text)
                except Exception as e:
                    logging.warning(f"Failed to write cache for {url}: {e}")
            return resp

        elif status == 429:
            # 优先尊重 Retry-After
            retry_after = resp.headers.get('Retry-After')
            if retry_after:
                try:
                    wait_seconds = int(retry_after)
                except ValueError:
                    wait_seconds = 60
            else:
                wait_seconds = min(60, (2 ** attempt) + random.uniform(0, 3))

            logging.warning(f"429 Too Many Requests for {url}. Waiting {wait_seconds}s (attempt {attempt}).")
            time.sleep(wait_seconds)
            attempt += 1
            continue

        elif 500 <= status < 600:
            backoff = (2 ** attempt) + random.random()
            logging.warning(f"Server error {status} for {url}. Backing off {backoff}s (attempt {attempt}).")
            time.sleep(backoff)
            attempt += 1
            continue

        else:
            logging.error(f"Failed to fetch {url}: HTTP {status}")
            return None

    logging.error(f"Max retries exceeded for {url}")
    return None

# ---------------------
# CSV 加载与过滤
# ---------------------
def load_and_filter_journals(csv_file, journal_level, selected_journals=None, selected_categories=None):
    """
    Filter journals based on the CCF level, selected journals, and categories
    """
    ccf_levels = ['A', 'B', 'C']
    df = pd.read_csv(csv_file)

    # Filter by journal level
    if journal_level in ccf_levels:
        valid_levels = ccf_levels[:ccf_levels.index(journal_level) + 1]
    else:
        logging.warning(
            f"Invalid journal level: {journal_level}. Using all levels.")
        valid_levels = ccf_levels

    filtered_df = df[df['级别'].isin(valid_levels)]

    # Filter by selected journals
    if selected_journals:
        selected_journals = [abbr.strip().lower() for abbr in selected_journals]  # Convert to lowercase
        filtered_df = filtered_df[filtered_df['刊物简称'].str.lower().isin(
            selected_journals)]

    # Filter by selected categories
    if selected_categories:
        selected_categories = [str(cat).strip() for cat in selected_categories]
        filtered_df.loc[:, '序号'] = filtered_df['序号'].astype(
            str).str.strip()
        filtered_df = filtered_df[filtered_df['序号'].isin(selected_categories)]

    return filtered_df[['刊物简称', '刊物全称', '类型', '网址']].values.tolist()
    """
    Filter journals based on the CCF level, selected journals, and categories
    """
    ccf_levels = ['A', 'B', 'C']
    df = pd.read_csv(csv_file)

    # Filter by journal level
    if journal_level in ccf_levels:
        valid_levels = ccf_levels[:ccf_levels.index(journal_level) + 1]
    else:
        logging.warning(
            f"Invalid journal level: {journal_level}. Using all levels.")
        valid_levels = ccf_levels

    filtered_df = df[df['级别'].isin(valid_levels)]

    # Filter by selected journals
    if selected_journals:
        selected_journals = [abbr.strip().lower()
                             for abbr in selected_journals]  # Convert to lowercase
        filtered_df = filtered_df[filtered_df['刊物简称'].str.lower().isin(
            selected_journals)]

    # Filter by selected categories
    if selected_categories:
        # Convert to string and strip
        selected_categories = [str(cat).strip() for cat in selected_categories]
        filtered_df.loc[:, '序号'] = filtered_df['序号'].astype(
            str).str.strip()  # Ensure '序号' is string and strip
        filtered_df = filtered_df[filtered_df['序号'].isin(selected_categories)]

    return filtered_df[['刊物简称', '刊物全称', '类型', '网址']].values.tolist()

# ---------------------
# 抓取与解析逻辑（将 session 显式传入）
# ---------------------
def fetch_papers(session, abbr, name, journal_type, baseurl, years, keywords):
    fetcher = {
        'conference': fetch_conference_papers,
        'journal': fetch_journal_papers
    }.get(journal_type.lower())

    if fetcher:
        return fetcher(session, abbr, name, baseurl, years, keywords)
    else:
        logging.error(f"Unknown type: {journal_type}")
        return []

def fetch_conference_papers(session, abbr, name, baseurl, years, keywords):
    return fetch_from_dblp(session, abbr, name, baseurl, years, keywords, entry_type='entry inproceedings')

def fetch_journal_papers(session, abbr, name, baseurl, years, keywords):
    return fetch_from_dblp(session, abbr, name, baseurl, years, keywords, entry_type='entry article', volume_pattern=True)

def fetch_from_dblp(session, abbr, name, baseurl, years, keywords, entry_type, volume_pattern=False):
    papers = []
    max_retries = 3  # 对页面本身的重试逻辑由 fetch_with_backoff 处理

    for year in years:
        logging.info(f"Fetching papers in {abbr} ({year})...")
        response = fetch_with_backoff(session, baseurl)
        if not response:
            logging.error(f"Failed to fetch base page for {abbr} ({year}), skipping.")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        if volume_pattern:
            pattern = re.compile(rf"Volume\s+\d+[:,]\s*{year}")
            link_tag = soup.find('a', string=pattern)
            if not link_tag:
                logging.warning(f"No data for {year} in {abbr}")
                continue
            url = link_tag['href'] if volume_pattern else baseurl
            logging.info(f"Search in {url}...")
            year_papers = parse_paper_entries(session, url, abbr, name, year, keywords, entry_type)
            papers += year_papers
            logging.info(f"Year {year}: Found {len(year_papers)} papers.")

        else:
            h2_tag = soup.find('h2', {'id': str(year)})
            link_tag = h2_tag.find_next('ul', class_='publ-list') if h2_tag else None
            if not link_tag:
                logging.warning(f"No data for {year} in {abbr}")
                continue
            navs = link_tag.find_all('nav', class_='publ')
            year_papers = []
            for nav in navs:
                # 原始代码在这里可能有链路解析的假设，做容错处理
                try:
                    li = nav.find_next('li', class_='drop-down')
                    a = li.find_next('a') if li else None
                    url = a['href'] if a and a.has_attr('href') else None
                except Exception:
                    url = None

                if not url:
                    logging.debug(f"Could not find article link under nav for {abbr} year {year}, skipping nav.")
                    continue

                logging.info(f"Search in {url}...")
                year_papers += parse_paper_entries(session, url, abbr, name, year, keywords, entry_type)
            papers += year_papers
            logging.info(f"Year {year}: Found {len(year_papers)} papers.")

    return papers

def parse_paper_entries(session, url, abbr, name, year, keywords, entry_type):
    response = fetch_with_backoff(session, url)
    if not response:
        logging.error(f"Failed to fetch paper entries page: {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    papers = []

    # DBLP 的条目 class 经常是 'entry inproceedings' 或 'entry article'
    for entry in soup.find_all('li', class_=entry_type):
        title_tag = entry.find('span', class_='title')
        if not title_tag:
            continue

        title = title_tag.text.strip()
        # 尝试获取 DOI 或链接（尽量容错）
        doi = None
        try:
            doi_link = entry.find_next('nav', class_='publ').find_next('li', class_='drop-down').find_next('a')
            if doi_link and doi_link.has_attr('href'):
                doi = doi_link['href']
        except Exception:
            doi = None

        for keyword in keywords:
            if keyword.lower() in title.lower():
                papers.append({
                    'Name': name,
                    'Abbreviation': abbr,
                    'Type': 'Conference' if 'inproceedings' in entry_type else 'Journal',
                    'Year': year,
                    'Keyword': keyword,
                    'Title': title,
                    'DOI': doi
                })
                break
    return papers

# ---------------------
# 工具函数
# ---------------------
def sanitize_sheet_name(sheet_name):
    """Replace invalid characters in sheet name with underscores."""
    return re.sub(r'[\\/*?:\[\]]', '_', sheet_name)

# ---------------------
# 主流程
# ---------------------
def main(csv_file_path, journal_level, num_years, keywords, selected_journals, selected_categories):
    # 创建 session（全局复用）
    session = create_session(retries=3, backoff_factor=0.5)

    current_year = datetime.datetime.now().year
    years_to_search = list(range(current_year - num_years + 1, current_year + 1))

    logging.info("----------------------------------------------------")
    logging.info(f"Searching for papers from the following parameters:")
    logging.info(f"Journal level: {journal_level}")
    logging.info(f"Years: {years_to_search}")
    logging.info(f"Keywords: {', '.join(keywords)}")
    if selected_journals:
        logging.info(f"Selected journals/conferences: {', '.join(selected_journals)}")
    else:
        logging.info("All journals/conferences will be included.")
    if selected_categories:
        logging.info(f"Selected categories: {', '.join(selected_categories)}")
    else:
        logging.info("All categories will be included.")
    logging.info("----------------------------------------------------")

    journal_list = load_and_filter_journals(csv_file_path, journal_level, selected_journals, selected_categories)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "papers_results.xlsx")

    if os.path.exists(output_file):
        os.remove(output_file)

    total_count = 0
    total_journals = len(journal_list)

    for idx, journal in enumerate(journal_list, start=1):
        journal_abbr, journal_name, journal_type, journal_url = journal
        logging.info("----------------------------------------------------")
        logging.info(f"Processing journal {idx}/{total_journals}")
        logging.info(f"Currently find {total_count} papers.")
        logging.info("----------------------------------------------------")
        logging.info(f"Fetching papers from {journal_abbr} ({journal_url})")

        papers = fetch_papers(session, journal_abbr, journal_name, journal_type, journal_url, years_to_search, keywords)

        if papers:
            papers_df = pd.DataFrame(papers)
            journal_count = len(papers_df)
            total_count += journal_count
            logging.info(f"Found {journal_count} papers in {journal_abbr}.")

            sanitized_sheet_name = sanitize_sheet_name(journal_abbr)

            if os.path.exists(output_file):
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    papers_df.to_excel(writer, index=False, sheet_name=sanitized_sheet_name)
            else:
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                    papers_df.to_excel(writer, index=False, sheet_name=sanitized_sheet_name)
        else:
            logging.info(f"No papers found in {journal_abbr}.")

    logging.info(f"Total papers found: {total_count}")
    logging.info(f"Results saved to {output_file}")

# ---------------------
# CLI 入口
# ---------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fetch papers from CCF journals and conferences.")
    parser.add_argument('--csv', type=str, default='./ccf2022.csv',
                        help='Path to the CCF journal/conference CSV file.')
    parser.add_argument('--level', type=str, default='C',
                        help='CCF rank to filter (e.g., A, B, C).')
    parser.add_argument('--years', type=int, default=3,
                        help='Number of recent years to search.')
    parser.add_argument('--keywords', type=str, required=True,
                        help='Comma-separated keywords to search.')
    parser.add_argument('--journals', type=str, default=None,
                        help='Comma-separated journal/conference abbreviations to filter (e.g., tc,tpds,isca).')
    parser.add_argument('--categories', type=str, default=None,
                        help='Comma-separated categories (序号) to filter. If not specified, all categories will be included.')

    args = parser.parse_args()

    keywords_list = [kw.strip().replace('_', ' ') for kw in args.keywords.split(',')]
    journals_list = [abbr.strip().replace('_', ' ') for abbr in args.journals.split(',')] if args.journals else None
    categories_list = [cat.strip() for cat in args.categories.split(',')] if args.categories else None

    main(args.csv, args.level, args.years, keywords_list, journals_list, categories_list)