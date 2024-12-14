# Curse Words Moderation Analysis Project

## Table of Contents

- Introduction
- Objectives
- Project Structure
- Installation and Setup
- How to Use
- Modules and Functions
  - Loaders
  - Models
  - Curate
  - Analysis
- Libraries Used
- Expected File Formats
- Results
- Notes and Observations
- Contributing
- License

## Introduction

This project aims to analyze how different language models interact with curse words in Hindi and Marathi. The motivation behind this study stemmed from a casual discussion about language models' handling of profanity in less commonly tested languages. By evaluating the responses of various models to such inputs, we hope to gain insights into their moderation capabilities and potential biases.

## Objectives

- To evaluate how different language models respond to curse words in Hindi and Marathi.
- To assess the moderation capabilities of these models concerning profanity in less commonly represented languages.
- To collect and analyze data on model responses for further studies or improvements.

## Project Structure

The project is organized into several Python modules, each responsible for specific functionalities:

```asx
├── .venv
│
├── project
│   ├── dataset
│   │   ├── loaders
│   │   │   ├── __init__.py
│   │   │   ├── curse_words.html
│   │   │   ├── curse_words.txt
│   │   │   └── loaders.py
│   │   │
│   │   ├── results
│   │   │   ├── analysis
│   │   │   │   ├── __init__.py
│   │   │   │   └── 20241118_15...
│   │   │   │
│   │   │   ├── dataset
│   │   │   │   ├── __init__.py
│   │   │   │   ├── result_202411...
│   │   │   │   ├── result_202411...
│   │   │   │   ├── result_202411...
│   │   │   │   └── result_202411...
│   │   │   │
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── curate.py
│   │   │   └── models.py
│   │   │
│   │   └── results_analysis...
│   │
│   ├── .gitignore
│   │
│   └── README...
│
└── requirements...
```

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/yourusername/curse-words-moderation.git
cd curse-words-moderation
```

### Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Set Up API Keys

Ensure you have the necessary API keys for the language models you intend to use:

- **Mistral AI**: Set the `MISTRAL_API_KEY` environment variable.
- **OpenAI**: Set the `OPENAI_API_KEY` environment variable.
- **Anthropic**: Set the `ANTHROPIC_API_KEY` environment variable.
 environment variable.

You can set environment variables in your terminal:

```bash
export MISTRAL_API_KEY='your_mistral_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export ANTHROPIC_API_KEY='your_anthropic_api_key'
```

On Windows:

```cmd
set MISTRAL_API_KEY=your_mistral_api_key
set OPENAI_API_KEY=your_openai_api_key
set ANTHROPIC_API_KEY=your_anthropic_api_key
```

## How to Use

1. **Prepare the Data**: Use loaders.py to generate or update the dataset of curse words.

   ```bash
   python loaders.py
   ```

2. **Run the Curate Script**: Invoke the models and collect their responses.

    ```bash
    python curate.py
    ```

3. **Analyze the Results**: Process and analyze the collected data.

   ```bash
   python analysis.py
   ```

4. **View the Results**: Check the CSV results in the `analysis` directory.

## Modules and Functions


2. **Analyze the Results**: Process and analyze the collected data.
loaders.py

Handles loading and preparing the dataset of curse words.

3. **View the Results**: Check the `results` directory for CSV files containing the datasets and analysis.
load_or_create_df()

: Loads the existing `curse_words.csv` or creates a new DataFrame if it doesn't exist
-

add_to_final_ds(word, language)

### `loaders.py`

Handles loading and preparing the dataset of curse words.

html_to_csv()

: Parses an HTML file (`curse_words.html`) and extracts curse words to add them to the dataset.

#### Key Functions

- `load_or_create_df()`: Loads the existing `curse_words.csv` or creates a new DataFrame if it doesn't exist.
- `add_to_final_ds(word, language)`: Adds a new word and its language to the dataset.
- `txt_csv()`: Converts curse words from a text file (`curse_words.txt`) to a CSV format.
- `html_to_csv()`: Parses an HTML file (`curse_words.html`) and extracts curse words to add them to the dataset.

models.py

Defines data models and constants used across the project.

#### Key Classes

-

ClassificationObject

: Represents the categories and scores assigned by the moderation models
-

ClassificationResponse

#### Usage

: Encapsulates the response from the moderation API.
Running `loaders.py` will invoke `html_to_csv()` by default:

OPENAI

).

#### Constants

-

MISTRAL_MODELS

: List of Mistral AI model names
-

OPENAI_MODELS

: List of OpenAI model names
-

### `models.py`

Defines data models and constants used across the project.

: List of moderation-specific model names.

###

#### Key Classes

- `ClassificationObject`: Represents the categories and scores assigned by the moderation models.
- `ClassificationResponse`: Encapsulates the response from the moderation API.
- `Providers`: Enum listing the available providers (`MISTRAL`, `OPENAI`).
_construct_clients()

: Constructs instances of clients for all models
-

_invoke_client(client, word)

: Invokes a client with a word and returns the response
-

process_word_model(client, word, language)

: Processes a single word-model combination
-

process_curse_words(csv_file_path, result_csv_path, limit)

: Processes all curse words using threading to invoke models concurrently
-

main()

#### Constants

: The main entry point that starts the curate process.

- `MISTRAL_MODELS`: List of Mistral AI model names.
- `OPENAI_MODELS`: List of OpenAI model names.
- `ANTHROPIC_MODELS`: List of Anthropic model names.
- `MODERATION_MODELS`: List of moderation-specific model names.

setup_logging()

: Sets up the logging configuration
-

flatten_json(y)

: Flattens nested JSON objects for easier analysis
-

_read_results(results_file)

: Reads the results dataset from a CSV file
-

_invoke_client(text, provider)

### `curate.py`

Invokes the language models with the curse words and stores their responses
-

_process(provider, limit)

: Processes the dataset and performs moderation checks using a specified provider
-

- `initialize_result_csv(csv_file_path, models)`: Initializes the results CSV with the required columns.
- `add_to_result_df_in_memory(df, word, model, response, language, lock)`: Adds or updates the response of a model for a given word in the in-memory DataFrame.
- `load_curse_words(csv_file_path)`: Loads the curse words dataset from a CSV file.
- `_construct_clients()`: Constructs instances of clients for all models.
- `_invoke_client(client, word)`: Invokes a client with a word and returns the response.
- `process_word_model(client, word, language)`: Processes a single word-model combination.
- `process_curse_words(csv_file_path, result_csv_path, limit)`: Processes all curse words using threading to invoke models concurrently.
- `main()`: The main entry point that starts the curate process.

: For parallel processing
  -

threading

: For thread-safe operations.

- `datetime`: For timestamping files and logs.
-

enum

: For enumerations.

- **Third-Party Libraries**
  -

pandas

: Data manipulation and analysis.

- `numpy`: Numerical computing (required by pandas).
-

pydantic

: Data validation using Python type hints
  -

bs4

 (BeautifulSoup): Parsing HTML documents.

- `fuzzywuzzy`: For string matching and similarity (if used).

- `langchain`: For interacting with language models
    -

langchain_openai

: OpenAI integration.
    -

langchain_mistralai

### `analysis.py`

Analyzes the responses collected from the language models.

**Installation**: All the required libraries are specified in the

requirements.txt

 file and can be installed using `pip install -r requirements.txt`.

- `setup_logging()`: Sets up the logging configuration.
- `flatten_json(y)`: Flattens nested JSON objects for easier analysis.
- `_read_results(results_file)`: Reads the results dataset from a CSV file.
- `_invoke_client(text, provider)`: Invokes the moderation API client based on the specified provider.
- `process_word_model(id, word, language, model, model_response, provider)`: Processes a single word-model combination to extract moderation data.
- `_process(provider, limit)`: Processes the dataset and performs moderation checks using a specified provider.
- `main()`: The main entry point that starts the analysis process.

in `curse_words.csv`
  -

word

: The curse word
  -

language

: The language of the word.

- Model response columns: Each model used will have a column containing the model's response to the word.

### HTML Files (`curse_words.html`)

- An HTML file containing a table of curse words.
- The words should be in the second `<td>` element of each `<tr>` (excluding the header row).

### Text Files (`curse_words.txt`)

- A plain text file with one curse word per line.
- Used by

txt_csv()

 function in

loaders.py

 to convert into a CSV format.

## Results

*This section will be updated with detailed results and analysis after the project execution is complete.*
**Note**: By default, the script uses `Providers.OPENAI` for analysis. Modify the `provider` argument in `_process` to change the moderation API provider.

*Specify the project license here, e.g., MIT License.*

---

*Disclaimer: This project is intended for research and analysis purposes. The use of profanity and curse words is solely for understanding language model behavior and does not reflect the views or intentions of the contributors.*- **Python Standard Libraries**:

- `os`: For interacting with the operating system.
- `logging`: For logging messages to the console and files.
- `concurrent.futures`: For parallel processing.
- `threading`: For thread-safe operations.
- `datetime`: For timestamping files and logs.
- `enum`: For enumerations.
- **Third-Party Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `numpy`: Numerical computing (required by pandas).
  - `pydantic`: Data validation using Python type hints.
  - `bs4` (BeautifulSoup): Parsing HTML documents.
  - `fuzzywuzzy`: For string matching and similarity (if used).
  - `langchain`: For interacting with language models.
    - `langchain_openai`: OpenAI integration.
    - `langchain_mistralai`: Mistral AI integration.
    - `langchain_anthropic`: Anthropic AI integration.

### Curse Words Dataset (`curse_words.csv`)

- **Columns**:
  - `id`: Integer identifier for each word.
  - `word`: The curse word.
  - `language`: The language of the word (e.g., 'hindi', 'marathi').

### Results Dataset (`result.csv`)

- **Columns**:
  - `id`: Corresponds to the `id` in `curse_words.csv`.
  - `word`: The curse word.
  - `language`: The language of the word.
  - Model response columns: Each model used will have a column containing the model's response to the word.
