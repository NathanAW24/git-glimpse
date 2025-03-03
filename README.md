# README

## Overview

### Gindex, a Superior Git Index for Code-based Search Query

This project aims to create a Retrieval-Augmented Generation (RAG) application that processes pull request (PR) data and uses a language model (LLM) to summarize the information. The goal is to develop a search engine that accurately retrieves PR data and provides concise summaries.

## Project Structure

The project is organized as follows:

```
.
├── 0_backups/
├── 1_data_collection/
├── 2_data_cleaning/
│   ├── convert_to_docs.py
│   └── pr_data/
├── 3_retriever/
│   └── basic/
│       └── retriever.py
├── 4_synthetic_data_generation/
├── 5_question_answering/
├── final_workflow_ui/
├── README.md
├── requirements.txt
```

### Folder Descriptions

- **0_backups/**: Backup files and data.
- **1_data_collection/**: Scripts and tools for collecting PR data using the GitHub API.
- **2_data_cleaning/**: Scripts for cleaning and processing PR data.
  - **convert_to_docs.py**: Converts PR data from JSON to text format.
  - **pr_data/**: Folder containing raw PR data files.
- **3_retriever/**: Scripts for setting up the RAG retriever.
  - **basic/**: Basic implementation of the retriever.
    - **retriever.py**: Sets up the RAG pipeline.
- **4_synthetic_data_generation/**: Scripts for generating synthetic queries using LLMs.
- **5_question_answering/**: Scripts for implementing zero-shot, few-shot, and other question-answering techniques.
- **final_workflow_ui/**: UI components for the final workflow.

## Requirements

The project requires the following Python packages, as specified in 

requirements.txt

:

```
langchain
langchain-community
langchain-chroma
transformers
chromadb
```

## Setting Up the Virtual Environment

### Step 1: Create a Virtual Environment

1. **Navigate to your project directory**:
   ```bash
   cd /path/to/your/project
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **On Linux/Mac**:
     ```bash
     source env_name/bin/activate
     ```
   - **On Windows**:
     ```bash
     .\env_name\Scripts\activate
     ```

### Step 2: Install the Required Libraries

1. **Create a 

requirements.txt

 file** (if it doesn't exist yet):
   ```txt
   langchain
   langchain-community
   langchain-chroma
   transformers
   chromadb
   ```

2. **Install the libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   pip list
   ```

### Step 3: Share with Your Friend

1. **Share your project directory**:
   Include the 

requirements.txt

 file but exclude the `env_name` directory (virtual environment folder) to keep it lightweight.

2. **Instruct your friend to replicate the setup**:
   - Clone/download the project folder.
   - Create and activate a virtual environment as shown in Step 1.
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### Step 4: Test the Environment

Run a Python script or a notebook in the environment to ensure everything works as expected:
```bash
python your_script.py
```

## Data Collection

In the data collection phase, we use the GitHub API to scrape PR data in JSON format. The data is stored in the `pr_data` folder.

## Data Cleaning

In the data cleaning phase, we convert the JSON data to text files using the `convert_to_docs.py` script. This script processes the JSON files and converts each PR into a document format suitable for RAG.

## Synthetic Data Generation

In the synthetic data generation phase, we use LLMs to generate synthetic queries based on the context in the processed documents. This helps in creating a more robust dataset for training and evaluation.

## Question Answering

In the question answering phase, we implement zero-shot, few-shot, and other techniques to answer questions based on the PR data. We also use citation to ensure the answers are well-supported by the context.

## Conclusion

This project demonstrates how to create a RAG application that processes PR data and uses an LLM to summarize the information. By following the steps outlined in this README, you can set up the project, process the data, and perform queries to retrieve and summarize PR information.

## GIthub

### Github v1
- without PR desc
- Broken reviews -> i used reviewsThreads instead of reviews

### Github v2
- with PR desc
- use reviews instead of reviewsThreads