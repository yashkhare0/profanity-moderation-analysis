import sys
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from datetime import datetime

from setup import setup_logging

CURSE_WORDS_CSV = Path('dataset') / f'curse_words_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
COLUMNS = ['id', 'word', 'language']
lock = RLock()
logger = setup_logging()

def load_or_create_dataset():
    """Load or create the dataset CSV file."""
    try:
        return pd.read_csv(CURSE_WORDS_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=COLUMNS)

def save_dataset(df):
    """Save the dataset to CSV file."""
    try:
        logger.info("Starting to save dataset...")
        with lock:
            logger.info(f"Saving {len(df)} words to {CURSE_WORDS_CSV}...")
            df.to_csv(CURSE_WORDS_CSV, index=False)
            logger.info("Dataset saved successfully.")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        raise e
    
def add_words_to_dataset(words_df):
    """Add multiple words to the dataset."""
    try:
        logger.info("Starting to add words to dataset...")
        with lock:
            logger.info("Loading existing dataset...")
            df = load_or_create_dataset()
            logger.info(f"Current dataset has {len(df)} words")
            logger.info("Checking for new words...")
            existing_words = set(df['word'].values)
            new_words = words_df[~words_df['word'].isin(existing_words)]
            logger.info(f"Found {len(new_words)} potential new words")
            if not new_words.empty:
                logger.info("Generating IDs for new words...")
                start_id = df['id'].max() + 1 if not df.empty else 1
                new_words['id'] = range(start_id, start_id + len(new_words))
                logger.info("Merging new words with existing dataset...")
                df = pd.concat([df, new_words], ignore_index=True)
                logger.info("Saving updated dataset...")
                save_dataset(df)
                logger.info(f"Added {len(new_words)} new words to the dataset.")
            else:
                logger.info("No new words to add.")
    except Exception as e:
        logger.error(f"Error adding words to dataset: {e}")
        raise e
    
def convert_txt_to_csv(txt_path: Path):
    """Convert a TXT file to the dataset.
    The TXT file should contain one word per line.
    """
    try:
        logger.info(f"Starting to process TXT file: {txt_path}")
        logger.info("Reading file contents...")
        words = pd.read_fwf(txt_path, names=['word'])
        logger.info(f"Found {len(words)} words in file")
        logger.info("Setting language to 'marathi'...")
        words['language'] = 'marathi'
        logger.info("Removing duplicates and empty values...")
        words = words.drop_duplicates().dropna()
        logger.info(f"After cleanup: {len(words)} unique words remaining")
        logger.info("Adding words to main dataset...")
        add_words_to_dataset(words)
        logger.info(f"Successfully completed processing TXT file: {txt_path}")
    except Exception as e:
        logger.error(f"Error processing TXT file {txt_path}: {e}")
        raise e

def convert_html_to_csv(path: Path = Path("curse_words.html")):
    """Convert an HTML file to the dataset."""
    try:
        logger.info(f"Starting to process HTML file: {path}")
        logger.info("Reading HTML file contents...")
        html_path = path
        with html_path.open('r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        logger.info("Extracting words from HTML table...")
        words_list = []
        rows = soup.find_all('tr')[1:]
        logger.info(f"Found {len(rows)} rows in the table")
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                word = cols[1].get_text(strip=True)
                words_list.append(word)
        logger.info(f"Extracted {len(words_list)} words from table")
        logger.info("Creating DataFrame and cleaning data...")
        words_df = pd.DataFrame(words_list, columns=["word"]).drop_duplicates().dropna()
        logger.info(f"After cleanup: {len(words_df)} unique words remaining")
        logger.info("Setting language to 'hindi'...")
        words_df['language'] = 'hindi'
        logger.info("Adding words to main dataset...")
        add_words_to_dataset(words_df)
        logger.info(f"Successfully completed processing HTML file: {path}")
    except Exception as e:
        logger.error(f"Error converting HTML to CSV: {e}")
        raise e

def process_files_in_directory(directory: Path):
    """Process all relevant files in a directory."""
    with ThreadPoolExecutor() as executor:
        for file in directory.iterdir():
            if file.suffix == ".html":
                executor.submit(convert_html_to_csv, file)
            elif file.suffix == ".txt":
                executor.submit(convert_txt_to_csv, file)

if __name__ == "__main__":
    base_path = Path("raw")
    logger.info(f"Processing files in directory: {base_path}")
    try:
        process_files_in_directory(base_path)
    except Exception as e:
        logger.error(f"Critical error encountered: {e}")
        sys.exit(1)
