import pandas as pd
from bs4 import BeautifulSoup # type: ignore


def add_to_final_ds(word, language):
    try:
        curse_words_df = pd.read_csv('curse_words.csv')
    except FileNotFoundError:
        curse_words_df = pd.DataFrame(columns=['id', 'word', 'language'])    
    if not curse_words_df.empty:
        max_id = curse_words_df["id"].max()
    else:
        max_id = 0
    new_row = {'id': max_id + 1, 'word': word, 'language': language}
    curse_words_df = pd.concat([curse_words_df, pd.DataFrame([new_row])], ignore_index=True)
    curse_words_df.to_csv('curse_words.csv', index=False)
    print(f"Added {word} to curse words dataset")

def process_curse_words():
    print("Converting curse words to CSV...")
    columns = ['word', 'language']
    df = pd.read_fwf('curse_words.txt', names=columns)
    df['language'] = 'marathi'
    df['id'] = range(1, len(df) + 1)
    df = df[['id', 'word', 'language']]
    df.to_csv('curse_words.csv', index=False)
    print("Conversion completed successfully!")

def html_to_csv():
    print("Converting HTML to CSV...")
    file_path = "curse_words.html"
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    devanagari_words = [row.find_all('td')[1].text for row in soup.find_all('tr')[1:]]
    df = pd.DataFrame(devanagari_words, columns=["Devanagari"])
    df = df.drop_duplicates()
    for word in df["Devanagari"]:
        add_to_final_ds(word, 'hindi')
    print(f"Added {len(df)} words to curse words dataset")
    
if __name__ == "__main__":
    html_to_csv()