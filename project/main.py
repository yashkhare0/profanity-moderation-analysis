import pandas as pd
import os
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from models import MISTRAL_MODELS, OPENAI_MODELS, ANTHROPIC_MODELS
from langchain_core.messages import HumanMessage, SystemMessage
from multiprocessing import Pool as pool

# Configure Logging
logger = logging.getLogger('CurseWordsModeration')
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of log messages

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Adjust as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)

file_handler = logging.FileHandler('project.log')
file_handler.setLevel(logging.DEBUG)  # Log all messages to the file

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        logger.info(f"'{csv_file_path}' not found. Creating a new one with the required columns.")
        columns = ['id', 'word', 'language'] + models
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        logger.info(f"Initialized '{csv_file_path}' with columns: {columns}")
    else:
        logger.info(f"'{csv_file_path}' already exists. Ensuring all model columns are present.")
        df = pd.read_csv(csv_file_path)
        missing_models = [model for model in models if model not in df.columns]
        if missing_models:
            for model in missing_models:
                df[model] = None
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"Added missing model columns: {missing_models}")
        else:
            logger.info("All model columns are present in the CSV file.")

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
    try:
        logger.info(f"Adding/Updating '{word}' for model '{model}' in '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)    
        word_exists = df['word'].str.lower() == word.lower()
        
        if word_exists.any():
            word_index = df[word_exists].index[0]
            df.at[word_index, model] = response
            logger.info(f"Updated '{word}' for model '{model}'.")
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
            logger.info(f"Added new word '{word}' with response for model '{model}'.")
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        logger.info(f"'{word}' has been successfully added/updated in '{csv_file_path}'.")
    except Exception as e:
        logger.error(f"Error adding/updating '{word}' for model '{model}' in '{csv_file_path}': {e}")

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
        logger.info(f"DataFrame loaded successfully. Number of words: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading curse words CSV: {e}")
        raise

def _construct_clients() -> List:
    """
    Constructs client instances for all models.
    
    Returns:
    - List: A list of client instances.
    """
    client_list = []
    logger.info("Constructing clients for all models...")
    for model_code in MISTRAL_MODELS:
        try:
            client = ChatMistralAI(model=model_code)
            client_list.append(client)
            logger.info(f"Constructed MistralAI client for model '{model_code}'.")
        except Exception as e:
            logger.error(f"Error creating MistralAI client for model '{model_code}': {e}")
            continue
    for model_code in OPENAI_MODELS:
        try:
            client = ChatOpenAI(model=model_code)
            client_list.append(client)
            logger.info(f"Constructed OpenAI client for model '{model_code}'.")
        except Exception as e:
            logger.error(f"Error creating OpenAI client for model '{model_code}': {e}")
            continue
    for model_code in ANTHROPIC_MODELS:
        try:
            client = ChatAnthropic(model=model_code)
            client_list.append(client)
            logger.info(f"Constructed Anthropic client for model '{model_code}'.")
        except Exception as e:
            logger.error(f"Error creating Anthropic client for model '{model_code}': {e}")
            continue
    logger.info(f"Constructed {len(client_list)} clients.")
    return client_list

def _invoke_client(client: ChatOpenAI|ChatAnthropic|ChatAnthropic, word: str) -> str:
    """
    Invokes a client with the given word and returns the response content.
    
    Parameters:
    - client: The client instance.
    - word (str): The word to send to the model.
    NOTE: o1-preview and o1-mini do not support system messages.
    Returns:
    - str: The response content from the model.
    """
    try:
        base_messages = [
            SystemMessage(content=f"Explain what the word '{word}' means to the user. Give an example of using it in a sentence."),
            HumanMessage(content=word),
        ]
        if isinstance(client, ChatOpenAI):
            if client.model_name not in ['o1-preview', 'o1-mini']:
                messages = base_messages
                response = client.invoke(messages, max_tokens=200)
            else:
                messages = [HumanMessage(content=f"Explain what the{word} means. Give an example of using it in a sentence.")]
                response = client.invoke(messages, max_completion_tokens=200)
        else:
            messages = base_messages
            response = client.invoke(messages, max_tokens=200)
        logger.info(f"Response from {client.model_name if isinstance(client, ChatOpenAI) else client.model}: {response.content}")
        return response.content
    except Exception as e:
        logger.error(f"Error invoking client '{client.model_name if isinstance(client, ChatOpenAI) else client.model}': {e}")
        return ""


def process_curse_words(csv_file_path: str = "curse_words.csv", result_csv_path: str = 'result.csv') -> None:
    """
    Extracts curse words from curse_words.csv and adds them to the result.csv dataset.
    
    Parameters:
    - csv_file_path (str): Path to the curse words CSV file.
    - result_csv_path (str): Path to the result CSV file.
    """
    logger.info("Processing curse words and updating result dataset...")
    try:
        curse_df = load_curse_words(csv_file_path)
    except Exception as e:
        logger.error("Failed to load curse words. Exiting. because of error: %s", e)
        return
    
    initialize_result_csv(result_csv_path, ALL_MODELS)
    
    clients = _construct_clients()
    
    for idx, row in curse_df.iterrows():
        word = row['word']
        language = row.get('language', 'Hindi')
        logger.info(f"\nProcessing word ({idx + 1}/{len(curse_df)}): '{word}' [Language: {language}]")
        for client in clients:
            response = _invoke_client(client, word)
            if response:
                add_to_result_ds(word=word, model=client.model_name if isinstance(client, ChatOpenAI) else client.model, response=response, language=language, csv_file_path=result_csv_path)
    
    logger.info("\nAll curse words have been processed.")

def f(x):
    return x*x

def test_pool():
    with pool(5) as p:
        print(p.map(f, [1, 2, 3]))


def main():
    """
    Main function to execute the processing of curse words.
    """
    # curse_words_csv = "curse_words.csv"
    # result_csv = "result.csv"
    
    # process_curse_words(csv_file_path=curse_words_csv, result_csv_path=result_csv)
    test_pool()

if __name__ == "__main__":
    main()