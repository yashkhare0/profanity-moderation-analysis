import pandas as pd
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from models import MISTRAL_MODELS, OPENAI_MODELS, ANTHROPIC_MODELS
from langchain_core.messages import HumanMessage, SystemMessage

ALL_MODELS = MISTRAL_MODELS + OPENAI_MODELS + ANTHROPIC_MODELS

os.environ["MISTRAL_API_KEY"] = os.environ.get("MISTRAL_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

def initialize_result_csv(csv_file_path: str, models: List[str]) -> None:
    """
    Initializes the result.csv file with the required columns if it doesn't exist.
    
    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - models (List[str]): List of model codes to create as columns.
    """
    if not os.path.exists(csv_file_path):
        print(f"'{csv_file_path}' not found. Creating a new one with the required columns.")
        columns = ['id', 'word', 'language'] + models
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"Initialized '{csv_file_path}' with columns: {columns}")
    else:
        print(f"'{csv_file_path}' already exists. Ensuring all model columns are present.")
        df = pd.read_csv(csv_file_path)
        missing_models = [model for model in models if model not in df.columns]
        if missing_models:
            for model in missing_models:
                df[model] = None  # Initialize new model columns with None
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            print(f"Added missing model columns: {missing_models}")
        else:
            print("All model columns are already present.")

def add_to_result_ds(word: str, model: str, response: str, language: str, csv_file_path: str = 'result.csv') -> None:
    """
    Adds or updates the response of a specific model for a given word in the result.csv dataset.
    
    Parameters:
    - word (str): The word to add or update.
    - model (str): The model code whose response is being added.
    - response (str): The response from the model.
    - language (str): The language of the word.
    - csv_file_path (str): Path to the result CSV file.
    """
    print(f"Adding/updating '{word}' with model '{model}' in '{csv_file_path}'.")    
    df = pd.read_csv(csv_file_path)    
    word_exists = df['word'].str.lower() == word.lower()
    
    if word_exists.any():
        word_index = df[word_exists].index[0]
        df.at[word_index, model] = response
        print(f"Updated '{word}' for model '{model}'.")
    else:
        if not df.empty:
            max_id = df['id'].max()
        else:
            max_id = 0
        new_id = max_id + 1
        
        new_row = {col: None for col in df.columns}
        new_row['id'] = new_id
        new_row['word'] = word
        new_row['language'] = language
        new_row[model] = response
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Added new word '{word}' with response for model '{model}'.")
    
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"'{word}' has been successfully added/updated in '{csv_file_path}'.")

def load_curse_words(csv_file_path: str = 'curse_words.csv') -> pd.DataFrame:
    """
    Load and return the curse words dataset.
    
    Parameters:
    - csv_file_path (str): Path to the curse words CSV file.
    
    Returns:
    - pd.DataFrame: DataFrame containing curse words.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"DataFrame loaded successfully. Number of words: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def _construct_clients() -> List:
    """
    Constructs client instances for all models.
    
    Returns:
    - List: A list of client instances.
    """
    client_list = []
    print("Constructing clients...")
    for model in MISTRAL_MODELS:
        try:
            client = ChatMistralAI(model=model)
            client_list.append(client)
            print(f"Constructed MistralAI client for model '{model}'.")
        except Exception as e:
            print(f"Error creating MistralAI client for model '{model}': {e}")
            continue
    for model in OPENAI_MODELS:
        try:
            client = ChatOpenAI(model=model)
            client_list.append(client)
            print(f"Constructed OpenAI client for model '{model}'.")
        except Exception as e:
            print(f"Error creating OpenAI client for model '{model}': {e}")
            continue
    for model in ANTHROPIC_MODELS:
        try:
            client = ChatAnthropic(model=model)
            client_list.append(client)
            print(f"Constructed Anthropic client for model '{model}'.")
        except Exception as e:
            print(f"Error creating Anthropic client for model '{model}': {e}")
            continue
    print(f"Constructed {len(client_list)} clients.")
    return client_list

def _invoke_client(client: ChatOpenAI|ChatAnthropic|ChatAnthropic, word: str) -> str:
    """
    Invokes a client with the given word and returns the response content.
    
    Parameters:
    - client: The client instance.
    - word (str): The word to send to the model.
    
    Returns:
    - str: The response content from the model.
    """
    try:
        messages = [
            SystemMessage(content="Explain what the word means to the user. Give an example of using it in a sentence."),
            HumanMessage(content=word),
        ]
        response = client.invoke(messages, max_tokens=200)
        print(f"Response from {client.model}: {response.content}\n")
        return response.content
    except Exception as e:
        print(f"Error invoking client '{client.model}': {e}")
        return ""

def process_curse_words(csv_file_path: str = "curse_words.csv", result_csv_path: str = 'result.csv') -> None:
    """
    Extracts curse words from curse_words.csv and adds them to the result.csv dataset.
    
    Parameters:
    - csv_file_path (str): Path to the curse words CSV file.
    - result_csv_path (str): Path to the result CSV file.
    """
    print("Processing curse words and updating result dataset...")    
    try:
        curse_df = load_curse_words(csv_file_path)
    except Exception as e:
        print("Failed to load curse words. Exiting. because of error: %s", e)
        return
    
    initialize_result_csv(result_csv_path, ALL_MODELS)
    
    clients = _construct_clients()
    
    for idx, row in curse_df.iterrows():
        word = row['word']
        language = row.get('language', 'Hindi')
        print(f"\nProcessing word ({idx + 1}/{len(curse_df)}): '{word}' [Language: {language}]")
        for client in clients:
            response = _invoke_client(client, word)
            if response:
                add_to_result_ds(word=word, model=client.model, response=response, language=language, csv_file_path=result_csv_path)
    
    print("\nProcessing completed successfully!")

def main():
    """
    Main function to execute the processing of curse words.
    """
    curse_words_csv = "curse_words.csv"
    result_csv = "result.csv"
    
    process_curse_words(csv_file_path=curse_words_csv, result_csv_path=result_csv)

if __name__ == "__main__":
    main()