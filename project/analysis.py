# analysis.py
import logging
import os
from pydantic import BaseModel
from models import Providers
from mistralai import Mistral
from openai import OpenAI
import pandas as pd
import concurrent.futures

# Constants for API Keys
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def setup_logging():
    """
    Set up logging configuration.
    """
    logger = logging.getLogger('CurseWordsModeration')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler('project.log')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

logger = setup_logging()

def flatten_json(y):
    """
    Flatten a nested JSON object.
    """
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            for item in x:
                flatten(item, name)
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def _read_results(results_file: str = "result_20241113_143445.csv"):
    """
    Read the dataset from a CSV file.
    """
    df = pd.read_csv("results/dataset/" + results_file)
    logger.info(f"Read {len(df)} rows from the dataset.")
    logger.debug(df.head())
    return df

def _invoke_client(text: str, provider: str):
    """
    Invoke the moderation API client based on the provider.

    Parameters:
    - text (str): The text to be moderated.
    - provider (str): The provider name.

    Returns:
    - dict: The response from the moderation API.
    """
    try:
        if provider == Providers.MISTRAL:
            client = Mistral(api_key=MISTRAL_API_KEY)
            model = "mistral-moderation-latest"
            response = client.classifiers.moderate(
                model=model,
                inputs=[text]
            )
        elif provider == Providers.OPENAI:
            client = OpenAI(api_key=OPENAI_API_KEY)
            model = "omni-moderation-latest"
            response = client.moderations.create(
                model=model,
                input=text
            )
        else:
            logger.error(f"Unsupported provider: {provider}")
            return None
        logger.debug(f"Received response from {provider}: {response}")
        response_dict = response.model_dump() if isinstance(response, BaseModel) else response.model_dump()
        return response_dict
    except Exception as e:
        logger.error(f"Error invoking client for provider '{provider}': {e}")
        return None

def process_word_model(id, word, language, model, model_response, provider):
    """
    Process a single word-model combination.

    Parameters:
    - id: The ID of the word.
    - word: The word being processed.
    - language: The language of the word.
    - model: The model name.
    - model_response: The model's response for the word.
    - provider: The provider for the moderation API.

    Returns:
    - dict: The processed data for the word-model combination.
    """
    try:
        if pd.isnull(model_response) or not isinstance(model_response, str):
            logger.warning(f"No response for word '{word}' with model '{model}'. Skipping.")
            return None
        response_dict = _invoke_client(model_response, provider)
        if response_dict is None:
            return None
        if 'results' in response_dict and len(response_dict['results']) > 0:
            response_data = response_dict['results'][0]
        else:
            response_data = response_dict
        flattened_response = flatten_json(response_data)
        new_row = {
            'id': id,
            'word': word,
            'language': language,
            'model': model,
        }
        flattened_response.pop('model', None)
        flattened_response.pop('id', None)
        new_row.update(flattened_response)
        logger.debug(f"Processed word '{word}' with model '{model}'.")
        return new_row
    except Exception as e:
        logger.error(f"Error processing word '{word}' with model '{model}': {e}")
        return None

def _process(provider: str = Providers.OPENAI, limit: int = None):
    """
    Process the dataset and perform moderation checks.

    Parameters:
    - provider (str): The moderation API provider.
    - limit (int): The number of rows to process from the dataset.
    """
    df = _read_results()
    model_columns = [col for col in df.columns if col not in ['id', 'word', 'language']]
    df = df.head(limit) if limit else df
    results_list = []
    logger.info(f"Starting processing with provider '{provider}'...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, row in df.iterrows():
            logger.info(f"Processing row {index + 1}...")
            id = row['id']
            word = row['word']
            language = row['language']
            for model in model_columns:
                logger.debug(f"Processing word '{word}' with model '{model}'...")
                model_response = row[model]
                future = executor.submit(
                    process_word_model, id, word, language, model, model_response, provider
                )
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results_list.append(result)
    if results_list:
        results_df = pd.DataFrame(results_list)
        analysis_dir = "results/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        filename = f"{analysis_dir}/{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{provider}_analysis.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to '{filename}' with {len(results_df)} entries.")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    """
    Main entry point for the analysis script.

    Indepth Summary of the script:
    - Read the dataset from a CSV file.
    - Iterate over each row in the dataset.
    - For each row, iterate over each model column.
    - Invoke the moderation API client for the text in the model column.
    - Process the response from the moderation API.
    - Flatten the response and create a new row with the processed data.
    - Save the processed data to a new CSV file.
    
    """
    _process(provider=Providers.OPENAI)