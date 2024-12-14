import pandas as pd
from bs4 import BeautifulSoup # type: ignore
from pathlib import Path

CURSE_WORDS_CSV = 'curse_words.csv'
COLUMNS = ['id', 'word', 'language']

def load_or_create_df():
    try:
        return pd.read_csv(CURSE_WORDS_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=COLUMNS)

def add_to_final_ds(word, language):
    df = load_or_create_df()
    new_id = df['id'].max() + 1 if not df.empty else 1
    new_row = {'id': new_id, 'word': word, 'language': language}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CURSE_WORDS_CSV, index=False)
    print(f"Added {word} to curse words dataset")

def txt_csv():
    print("Converting curse words to CSV...")
    df = pd.read_fwf('curse_words.txt', names=['word', 'language'])
    df['language'] = 'marathi'
    df['id'] = range(1, len(df) + 1)
    df = df[COLUMNS]
    df.to_csv(CURSE_WORDS_CSV, index=False)
    print("Conversion completed successfully!")

def html_to_csv():
    print("Converting HTML to CSV...")
    html_path = Path("curse_words.html")
    
    with html_path.open('r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    words = pd.DataFrame(
        [row.find_all('td')[1].text for row in soup.find_all('tr')[1:]],
        columns=["Devanagari"]
    ).drop_duplicates()
    
    for word in words["Devanagari"]:
        add_to_final_ds(word, 'hindi')
    print(f"Added {len(words)} words to curse words dataset")

if __name__ == "__main__":
    html_to_csv()