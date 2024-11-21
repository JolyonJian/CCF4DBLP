import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime
import logging
import os
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("./output/fetch_papers.log",
                            mode='w')  # Output to file
    ]
)

# Load and filter journal information


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

# General function to fetch papers based on type (conference/journal)


def fetch_papers(abbr, name, journal_type, baseurl, years, keywords):
    fetcher = {
        'conference': fetch_conference_papers,
        'journal': fetch_journal_papers
    }.get(journal_type.lower())

    if fetcher:
        return fetcher(abbr, name, baseurl, years, keywords)
    else:
        logging.error(f"Unknown type: {journal_type}")
        return []

# Fetch conference papers


def fetch_conference_papers(abbr, name, baseurl, years, keywords):
    return fetch_from_dblp(abbr, name, baseurl, years, keywords, entry_type='entry inproceedings')

# Fetch journal papers


def fetch_journal_papers(abbr, name, baseurl, years, keywords):
    return fetch_from_dblp(abbr, name, baseurl, years, keywords, entry_type='entry article', volume_pattern=True)

# Core fetching logic with retry mechanism


def fetch_from_dblp(abbr, name, baseurl, years, keywords, entry_type, volume_pattern=False):
    papers = []
    max_retries = 10  # Maximum number of retries for each failed request

    for year in years:
        logging.info(f"Fetching papers in {abbr} ({year})...")
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(baseurl)
                if response.status_code != 200:
                    logging.error(
                        f"Failed to fetch page: {response.status_code}")
                    retries += 1
                    time.sleep(5)  # Wait before retrying
                    continue  # Retry fetching
                else:
                    break  # Success, break out of retry loop
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data: {e}")
                retries += 1
                time.sleep(5)  # Wait before retrying
                continue  # Retry fetching

        if retries == max_retries:
            logging.error(
                f"Max retries exceeded for {abbr} ({year}), skipping this year.")
            continue  # Skip to next year or journal

        soup = BeautifulSoup(response.text, 'html.parser')

        if volume_pattern:
            pattern = re.compile(rf"Volume\s+\d+[:,]\s*{year}")
            link_tag = soup.find('a', string=pattern)
            if not link_tag:
                logging.warning(f"No data for {year}")
                continue
            url = link_tag['href'] if volume_pattern else baseurl
            logging.info(f"Search in {url}...")
            year_papers = parse_paper_entries(
                url, abbr, name, year, keywords, entry_type)
            papers += year_papers
            logging.info(f"Year {year}: Found {len(year_papers)} papers.")

        else:
            h2_tag = soup.find('h2', {'id': str(year)})
            link_tag = h2_tag.find_next(
                'ul', class_='publ-list') if h2_tag else None
            if not link_tag:
                logging.warning(f"No data for {year}")
                continue
            navs = link_tag.find_all('nav', class_='publ')
            year_papers = []
            for nav in navs:
                url = nav.find_next(
                    'li', class_='drop-down').find_next('a')['href']
                logging.info(f"Search in {url}...")
                year_papers += parse_paper_entries(
                    url, abbr, name, year, keywords, entry_type)
            papers += year_papers
            logging.info(f"Year {year}: Found {len(year_papers)} papers.")

    return papers

# Parse paper entries from DBLP
def parse_paper_entries(url, abbr, name, year, keywords, entry_type):
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch paper entries: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    papers = []

    for entry in soup.find_all('li', class_=entry_type):
        title_tag = entry.find('span', class_='title')
        if not title_tag:
            continue

        title = title_tag.text.strip()
        doi = next((a['href'] for a in entry.find_all(
            'a', href=True) if 'doi' in a['href']), '')

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

# Replace invalid characters in sheet names
def sanitize_sheet_name(sheet_name):
    """Replace invalid characters in sheet name with underscores."""
    return re.sub(r'[\\/*?:\[\]]', '_', sheet_name)

# Main logic
def main(csv_file_path, journal_level, num_years, keywords, selected_journals, selected_categories):
    current_year = datetime.datetime.now().year
    years_to_search = list(
        range(current_year - num_years + 1, current_year + 1))

    # Output the current search parameters
    logging.info("----------------------------------------------------")
    logging.info(f"Searching for papers from the following parameters:")
    logging.info(f"Journal level: {journal_level}")
    logging.info(f"Years: {years_to_search}")
    logging.info(f"Keywords: {', '.join(keywords)}")
    if selected_journals:
        logging.info(
            f"Selected journals/conferences: {', '.join(selected_journals)}")
    else:
        logging.info("All journals/conferences will be included.")
    if selected_categories:
        logging.info(f"Selected categories: {', '.join(selected_categories)}")
    else:
        logging.info("All categories will be included.")
    logging.info("----------------------------------------------------")

    # Load and filter journals
    journal_list = load_and_filter_journals(
        csv_file_path, journal_level, selected_journals, selected_categories)

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "papers_results.xlsx")

    # Clear the Excel file if it exists, then create a new one
    if os.path.exists(output_file):
        os.remove(output_file)

    total_count = 0
    total_journals = len(journal_list)

    # Process each journal and fetch papers
    for idx, journal in enumerate(journal_list, start=1):
        journal_abbr, journal_name, journal_type, journal_url = journal
        logging.info("----------------------------------------------------")
        logging.info(f"Processing journal {idx}/{total_journals}")
        logging.info(f"Currently find {total_count} papers.")
        logging.info("----------------------------------------------------")
        logging.info(f"Fetching papers from {journal_abbr} ({journal_url})")

        papers = fetch_papers(journal_abbr, journal_name,
                              journal_type, journal_url, years_to_search, keywords)

        # Save papers for this journal to Excel without overwriting the entire file
        if papers:
            papers_df = pd.DataFrame(papers)
            journal_count = len(papers_df)
            total_count += journal_count
            logging.info(f"Found {journal_count} papers in {journal_abbr}.")

            # Handling invalid characters in worksheet names
            sanitized_sheet_name = sanitize_sheet_name(journal_abbr)

            if os.path.exists(output_file):
                # Load existing workbook to append data
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    papers_df.to_excel(writer, index=False,
                                       sheet_name=sanitized_sheet_name)
            else:
                # Create a new workbook
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                    papers_df.to_excel(writer, index=False,
                                       sheet_name=sanitized_sheet_name)
        else:
            logging.info(f"No papers found in {journal_abbr}.")

    logging.info(f"Total papers found: {total_count}")
    logging.info(f"Results saved to {output_file}")


# Add command line argument support
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

    # Convert input keywords and selected journals to lists
    keywords_list = [kw.strip() for kw in args.keywords.split(',')]
    journals_list = [abbr.strip() for abbr in args.journals.split(
        ',')] if args.journals else None
    categories_list = [cat.strip() for cat in args.categories.split(
        ',')] if args.categories else None

    # Run the main logic
    main(args.csv, args.level, args.years,
         keywords_list, journals_list, categories_list)
