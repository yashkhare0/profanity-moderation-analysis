import logging
import os
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel
from models import MISTRAL_MODELS, OPENAI_MODELS, ANTHROPIC_MODELS, ClassificationResponse, Providers
from mistralai import Mistral
from openai import OpenAI



logger = logging.getLogger('CurseWordsModeration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('project.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

ALL_MODELS = MISTRAL_MODELS + OPENAI_MODELS + ANTHROPIC_MODELS

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
        logger.debug(f"Initialized '{csv_file_path}' with columns: {columns}")
    else:
        logger.info(f"'{csv_file_path}' already exists. Ensuring all model columns are present.")
        try:
            df = pd.read_csv(csv_file_path, dtype=str)
            missing_models = [model for model in models if model not in df.columns]
            if missing_models:
                for model in missing_models:
                    df[model] = None
                df.to_csv(csv_file_path, index=False, encoding='utf-8')
                logger.info(f"Added missing model columns: {missing_models}")
            else:
                logger.info("All model columns are present in the CSV file.")
        except Exception as e:
            logger.error(f"Error initializing '{csv_file_path}': {e}")
            raise


def load_results(csv_file_path: str = 'result_20241113_143445.csv') -> pd.DataFrame:
    """
    Load and return the curse words dataset.
    
    Parameters:
    - csv_file_path (str): Path to the curse words CSV file.
    
    Returns:
    - pd.DataFrame: DataFrame containing curse words.
    """
    logger.info(f"Loading curse words from '{csv_file_path}'.")
    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"DataFrame loaded successfully. Number of words: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading '{csv_file_path}': {e}")
        raise


def _construct_pydantic(response_json: Dict[str, Any]) -> ClassificationResponse:
    """
    Construct a Pydantic model from the JSON representation of the moderation response.

    Parameters:
    - response_json (Dict[str, Any]): The JSON data from the moderation API response.

    Returns:
    - ClassificationResponse: The constructed Pydantic model.
    """
    logger.info("Constructing Pydantic model from JSON response.")
    logger.info(f"Response JSON: {type(response_json)}")
    try:
        pydantic_response = ClassificationResponse.model_validate(response_json)
        logger.info(f"Successfully constructed Pydantic model: {pydantic_response}")
        return pydantic_response
    except Exception as e:
        logger.error(f"Error constructing Pydantic model: {e}", exc_info=True)
        raise


def _invoke_client(text:str, provider:"str" =Providers.OPENAI):
    """
    Invoke the client based on the provider.
    
    Parameters:
    - provider (str): Provider name.
    
    Returns:
    - Client: Client object for the provider.
    """
    client = OpenAI()
    if provider == Providers.MISTRAL:
        client = Mistral(api_key=MISTRAL_API_KEY)
        model = "mistral-moderation-latest"
        response = client.classifiers.moderate(
            model = "mistral-moderation-latest",  
            inputs=[text]
        )
    elif provider == Providers.OPENAI:
        client = OpenAI()
        model="omni-moderation-latest"
        response = client.moderations.create(
            model=model,
            input=text
        )
    logger.info(f"Response from {provider}: {type(response)}")
    response_dict = response.model_dump() if isinstance(response, BaseModel) else response.model_dump()
    logger.info(f"Response from {provider}: {response_dict}")
    return _construct_pydantic(response_dict)

if __name__ == "__main__":
    _invoke_client("Madarchod", Providers.MISTRAL)