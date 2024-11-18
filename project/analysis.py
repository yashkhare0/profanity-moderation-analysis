import logging
import os
from pydantic import BaseModel
from models import MISTRAL_MODELS, OPENAI_MODELS, ANTHROPIC_MODELS, Providers
from mistralai import Mistral
from openai import OpenAI
import pandas as pd


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

def flatten_json(y):
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


def _read_results(results_file:str="result_20241113_143445.csv"):
    df = pd.read_csv("results/dataset/"+results_file)
    logger.info(df.head())
    return df


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
    return response_dict


def _process(provider: "str" = Providers.OPENAI, limit: int = 2):
    df = _read_results()
    results_list = []
    model_columns = [col for col in df.columns if col not in ['id', 'word', 'language']]
    df = df.head(limit)
    for index, row in df.iterrows():
        id = row['id']
        word = row['word']
        language = row['language']
        for model in model_columns:
            try:
                model_response = row[model]
                if pd.isnull(model_response) or not isinstance(model_response, str):
                    logger.warning(f"No response for word '{word}' with model '{model}'. Skipping.")
                    continue
                response_dict = _invoke_client(model_response,provider)
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
                results_list.append(new_row)
            except Exception as e:
                logger.error(f"Error processing word '{word}' with model '{model}': {e}")
                continue
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('results_analysis.csv', index=False)
    logger.info("Results saved to 'results_analysis.csv'")

if __name__ == "__main__":
    _process()
