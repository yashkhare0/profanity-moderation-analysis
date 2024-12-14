# Local Languages Profanity Responses Analysis : Hindi and Marathi

## Disclaimer

This project is intended for research and educational purposes. The inclusion of profanity is solely for analyzing language model behaviors in handling such content. The project does not endorse or promote the use of offensive language.

## Table of Contents

- Introduction
- Motivation
- Objectives
- Project Structure
- Installation and Setup
  - Prerequisites
  - Clone the Repository
  - Install Dependencies
  - Set Up API Keys
- How to Use
  - Quick Start
    1. Prepare the Data
    2. Run the Curate Script
    3. Analyze the Results
    4. View the Results
- Mistral Integration
- Modules and Packages
    1. `loaders.py`
        - Key Functions
    2. `models.py`
        - Key Classes and Constants
    3. `curate.py`
        - Key Functions and Classes
    4. `analysis.py`
        - Key Functions
- Expected File Formats
    1. Curse Words Dataset (`curse_words.csv`)
    2. Results Dataset (`result_YYYYMMDD_HHMMSS.csv`)
    3. HTML Files (`curse_words.html`)
    4. Text Files (`curse_words.txt`)
- Libraries Used
  - Python Standard Libraries
  - Third-Party Libraries
- Results
- Analysis and Observations
- Challenges Faced
- Contributing
- License
- Acknowledgements
- Ethical Considerations
- Future Work
- Testing and Continuous Integration
- Contact Information
- Disclaimer

---

## Introduction

This project aims to analyze how different language models interact with curse words in Hindi and Marathi. The motivation behind this study stems from a discussion about the handling of profanity by language models in less commonly tested languages. By evaluating the responses of various models to such inputs, we hope to gain insights into their moderation capabilities and potential biases.

## Motivation

Language models have become integral to a wide range of applications, from chatbots to automated content generation. However, the way these models handle profanity, especially in less commonly represented languages like Hindi and Marathi, remains underexplored. Understanding this behavior is crucial for:

- **Improving User Experience**: Ensuring that users are not exposed to inappropriate content.
- **Enhancing Model Safety**: Preventing the dissemination of harmful or offensive material.
- **Promoting Inclusivity**: Addressing potential biases that may affect speakers of different languages.

## Objectives

- **Evaluate Responses**: Assess how different language models respond to curse words in Hindi and Marathi.
- **Assess Moderation Capabilities**: Examine the effectiveness of these models in moderating profanity.
- **Identify Biases**: Detect any biases in the models' handling of different languages.
- **Data Collection and Analysis**: Collect comprehensive data on model responses for further studies or improvements.

## Project Structure

The project is organized into several Python modules and directories, each responsible for specific functionalities:

```bash
curse-words-moderation/
├── .venv/
├── project/
│   ├── dataset/
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── curse_words.html
│   │   │   ├── curse_words.txt
│   │   │   └── loaders.py
│   │   ├── results/
│   │   │   ├── analysis/
│   │   │   │   ├── __init__.py
│   │   │   │   └── analysis_results.csv
│   │   │   ├── dataset/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── result_YYYYMMDD_HHMMSS.csv
│   │   │   │   └── ...
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── curate.py
│   │   │   └── models.py
│   ├── .gitignore
│   └── README.md
├── requirements.txt
└── project.log
```

## Installation and Setup

### Prerequisites

- **Python 3.7 or higher**
- **Git**
- **API Keys** for the language models you intend to use:
  - Mistral AI
  - OpenAI
  - Anthropic

### Clone the Repository

```bash
git clone https://github.com/yashkhare0/profanity-moderation-analysis.git
cd profanity-moderation-analysis
```

### Install Dependencies

It is recommended to create a virtual environment to avoid package conflicts.

#### On macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Set Up API Keys

Set the necessary API keys as environment variables.

#### macOS/Linux

```bash
export MISTRAL_API_KEY='your_mistral_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export ANTHROPIC_API_KEY='your_anthropic_api_key'
```

#### Windows

```cmd
set MISTRAL_API_KEY=your_mistral_api_key
set OPENAI_API_KEY=your_openai_api_key
set ANTHROPIC_API_KEY=your_anthropic_api_key
```

Alternatively, you can use a `.env` file with [python-dotenv](https://pypi.org/project/python-dotenv/).

## How to Use

### Quick Start

```bash
# 1. Set API Keys as described above
# 2. Prepare data:
python loaders.py

# 3. Run the main curation:
python curate.py

# 4. Analyze results:
python analysis.py

# Check results in results/dataset and results/analysis directories.
```

### 1. Prepare the Data

```bash
python loaders.py
```

- Parses `curse_words.html` or `curse_words.txt`.
- Extracts curse words and languages.
- Saves data into `curse_words.csv`.

### 2. Run the Curate Script

```bash
python curate.py
```

- Loads curse words from `curse_words.csv`.
- Invokes each specified language model (including Mistral) with the curse words.
- Stores responses in `results/dataset/result_YYYYMMDD_HHMMSS.csv`.

### 3. Analyze the Results

```bash
python analysis.py
```

- Reads collected responses.
- Invokes moderation APIs to analyze model outputs.
- Saves processed results to `results/analysis/analysis_YYYYMMDD_HHMMSS.csv`.

### 4. View the Results

Check CSV files in `results/analysis`. Use Excel, Pandas, or Tableau for further analysis.

## Mistral Integration

This project is designed to work with multiple language models, including Mistral. The `models.py` file defines `MISTRAL_MODELS`, and the environment variable `MISTRAL_API_KEY` is required to invoke Mistral's public API or SDK. In the `curate.py` script, the `_construct_clients()` function creates clients for these models. By running `python curate.py` after setting the `MISTRAL_API_KEY`, you ensure that Mistral’s models are invoked alongside other providers.

If you have Mistral’s public endpoints or SDK, this integration will showcase how their models respond to curse words, allowing you to assess their moderation capabilities and compare them with other providers.

## Modules and Packages

### 1. `loaders.py`

**Key Functions**:

- `load_or_create_df()`
- `add_to_final_ds(word, language)`
- `txt_csv()`
- `html_to_csv()`

Usage:

```bash
python loaders.py
```

### 2. `models.py`

**Key Classes and Constants**:

- `ClassificationObject`
- `ClassificationResponse`
- `Providers` (Enum: `MISTRAL`, `OPENAI`, `ANTHROPIC`)

**Constants**:

- `MISTRAL_MODELS`
- `OPENAI_MODELS`
- `ANTHROPIC_MODELS`
- `MODERATION_MODELS`

### 3. `curate.py`

**Key Functions**:

- `initialize_result_csv()`
- `add_to_result_df_in_memory()`
- `load_curse_words()`
- `_construct_clients()`
- `_invoke_client()`
- `process_word_model()`
- `process_curse_words()`
- `main()`

Usage:

```bash
python curate.py
```

### 4. `analysis.py`

**Key Functions**:

- `setup_logging()`
- `flatten_json()`
- `_read_results()`
- `_invoke_client()`
- `process_word_model()`
- `_process()`
- `main()`

Usage:

```bash
python analysis.py
```

## Expected File Formats

### 1. Curse Words Dataset (`curse_words.csv`)

**Columns**:  

- `id`  
- `word`  
- `language`

### 2. Results Dataset (`result_YYYYMMDD_HHMMSS.csv`)

**Columns**:  

- `id`  
- `word`  
- `language`  
- Model-specific response columns

### 3. HTML Files (`curse_words.html`)

- Contains curse words in a table.
- Words are typically in the second `<td>` element.

### 4. Text Files (`curse_words.txt`)

- One curse word per line.

## Libraries Used

### Python Standard Libraries

- `os`
- `logging`
- `concurrent.futures`
- `threading`
- `datetime`
- `enum`
- `typing`
- `pathlib`

### Third-Party Libraries

- `pandas`
- `numpy`
- `pydantic`
- `beautifulsoup4 (bs4)`
- `langchain`
- `openai`
- `mistralai`
- `anthropic`

Install via:

```bash
pip install -r requirements.txt
```

## Results

After running the scripts:

- **Model Responses**: `results/dataset/`
- **Analysis Results**: `results/analysis/`

## Analysis and Observations

- **Model Effectiveness**
- **Language Differences**
- **Moderation Gaps**

## Challenges Faced

- Data collection complexity.
- API rate limits and response times.
- Addressing model biases.
- Ethical considerations in handling sensitive content.

## Contributing

1. Fork the repository via GitHub.
2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make changes and commit:

   ```bash
   git commit -m "Description of your changes"
   ```

4. Push to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request on GitHub.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Contributors
- OpenAI, Mistral AI, Anthropic Documentation
- Special thanks to library and API developers

## Ethical Considerations

- Responsible handling of profanity.
- Aim: Improve moderation and reduce harmful content.

## Future Work

- Expanded language support.
- Enhanced model improvement with AI providers.

## Testing and Continuous Integration

While this project does not currently include formal automated tests, you can test it by running the main scripts (`loaders.py`, `curate.py`, and `analysis.py`) and verifying the generated output files. Consider adding unit tests for data loading and response parsing if you plan to extend this project further. For continuous integration, setting up GitHub Actions or another CI platform to run tests and lint checks on every commit is recommended to maintain code quality over time.

## Contact Information

- Project Maintainer: [Yash Khare](https://github.com/yashkhare0)
- Email: <yash.khare.work@gmail.com>
