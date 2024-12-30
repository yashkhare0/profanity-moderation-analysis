# analyse.py
import os
import sys
from pydantic import BaseModel
from models import Providers
from mistralai import Mistral
from openai import OpenAI
import pandas as pd
from setup import setup_logging
import time
import random
import concurrent.futures

# Constants for API Keys
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

logger = setup_logging()

def flatten_json(y):
    """
    Flatten a nested JSON object.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error flattening JSON: {e}")
        return None

def _read_results():
    """
    take the latest results csv file from dataset folder
    Read the dataset from a CSV file.
    """
    try:
        dataset_dir = "results/dataset/"
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(dataset_dir, x)))
        latest_file = os.path.join(dataset_dir, latest_file)
        logger.info(f"Using latest results file: {latest_file}")
        if not latest_file.endswith(".csv"):
            logger.error(f"Invalid file format: {latest_file}")
            return None
        df = pd.read_csv(latest_file)
        logger.info(f"Read {len(df)} rows from the dataset.")
        logger.debug(df.head())
        return df
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return None

def _invoke_client(text: str, provider: str):
    """
    Invoke the moderation API client based on the provider.

    Parameters:
    - text (str): The text to be moderated.
    - provider (str): The provider name.

    Returns:
    - dict: The response from the moderation API.
    """
    max_retries = 5
    backoff_factor = 2
    initial_wait = 1  # in seconds

    for attempt in range(max_retries):
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
            response_dict = response.model_dump() if isinstance(response, BaseModel) else response
            return response_dict
        except Exception as e:
            if '429' in str(e):
                wait_time = initial_wait * (backoff_factor ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded for provider '{provider}'. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error invoking client for provider '{provider}': {e}")
                sys.exit(1)
    logger.error(f"Max retries exceeded for provider '{provider}'.")
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
        if pd.isnull(model_response):
            logger.warning(f"Model response is null for word '{word}' with model '{model}'. Skipping.")
            return None
        if not isinstance(model_response, str):
            logger.warning(f"Model response is not a string for word '{word}' with model '{model}'. Skipping.")
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
        raise e

def _process(provider: str = Providers.OPENAI, limit: int = None):
    """
    Process the dataset and perform moderation checks.

    Parameters:
    - provider (str): The moderation API provider.
    - limit (int): The number of rows to process from the dataset.
    """
    try:
        df = _read_results()
        model_columns = [col for col in df.columns if col not in ['id', 'word', 'language']]
        df = df.head(limit) if limit else df
        results_list = []
        logger.info(f"Starting processing with provider '{provider}'...")
        for index, row in df.iterrows():
            logger.info(f"Processing row {index + 1}...")
            id = row['id']
            word = row['word']
            language = row['language']
            for model in model_columns:
                logger.debug(f"Processing word '{word}' with model '{model}'...")
                model_response = row[model]
                result = process_word_model(id, word, language, model, model_response, provider)
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
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise e

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
    # try:
    #_process(Providers.MISTRAL)
    #     logger.info("Starting processing...")
    #     _process(Providers.OPENAI)
    # except Exception as e:
    #     logger.error(f"Error processing dataset: {e}")
    #     raise e
    try:
        logger.info("Starting processing...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_process, provider=Providers.MISTRAL),
                executor.submit(_process, provider=Providers.OPENAI)
            ]
            concurrent.futures.wait(futures)
            logger.info("Processing completed.")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise e