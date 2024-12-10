from dotenv import load_dotenv
import os
import re
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import json
from index import BM25Retriever, _read_documents_from_folder


# Load environment variables
load_dotenv()

# Set OpenAI API Key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

# Step 1: Define your questions for evaluation
questions = [
    "How to fix key warning in Dropdown component?",
    "How to fix modal button's \"onPress\" issue?",
    "How do I correctly import `CardFooter` in the `Card` component composition?",
    "How to fix typos in component stories?",
    "How can I automatically focus on the first non-disabled item in an autocomplete list when the list is rendered?",
    "How to fix footer blur issue on Firefox in React Card component?",
    "How do I automate package versioning and npm publishing using Changesets in a CI/CD pipeline?",
    "How do I correctly type the \"as\" prop for buttons in TypeScript?",
    "How to prevent focus on clear button when input is disabled?",
    "How do I automate package versioning and publishing using Changesets?",
    "How can I resolve circular dependencies between two React components, specifically when one component imports the other and vice versa? Are there any best practices or alternative approaches to avoid this issue, especially in a create-react-app environment?",
    "How can I clean up the code in a React component by removing unused import statements to improve code efficiency and maintainability? Specifically, what steps should I follow to identify and safely remove unused imports in a TypeScript file like `badge.tsx`?",
    "How should I correct a broken or outdated link in the documentation for our project to ensure users are directed to the correct and current resources? Can you provide an example of a previous bug fix where a documentation link was updated?",
    "How can I remove the focus styles from an input component without affecting its overall functionality? Specifically, what adjustments are necessary in the component's styling and structure to ensure the focus border is not displayed, and how should the component's code be refactored to maintain clean and efficient code?",
    "How can I modify the behavior of an autocomplete component so that the popover remains open after the input is cleared, and the input field regains focus correctly? Additionally, how can I ensure that these changes do not introduce a breaking change to the current functionality?",
    "How can we prevent React warnings about unrecognized non-DOM attributes like `isSelected`, `isIndeterminate`, and `disableAnimation` being passed to SVG elements in our checkbox component? Specifically, what changes are needed in the icon components to ensure these props are properly handled and not passed to the DOM?",
    "How can I ensure that the placeholder text for the Select component is displayed correctly when no items are selected, especially in a controlled component scenario? What changes should be made to the component's rendering logic to check if the state is empty and return the placeholder accordingly?",
    "What is the correct attribute name to use in the documentation for disabling the rotation of the selector icon in the Autocomplete and Select components? I've come across different references and want to ensure I'm using the correct one.",
    "What changes or updates were made in the `v2.4.2` release of the NextUI library, specifically focusing on bug fixes, feature additions, and modifications to existing components? Additionally, were there any breaking changes or updates to dependencies that I should be aware of when integrating this version into my project?",
    "How can I set up a GitHub Actions workflow to automate the versioning and prerelease publishing of packages using Changesets, ensuring that updates are applied when changes are merged into a specific branch, and how can I handle exiting prerelease mode if needed?"
]

# Step 2: Load vector database
folder_path = "processed_docs"  # Ensure the path matches your folder structure
documents = _read_documents_from_folder(folder_path)
retriever = BM25Retriever(documents)

# Step 3: Initialize the LLM judge using GPT-4


class RelevanceScore(BaseModel):
    score: int = Field(
        description="The relevance score of the retrieval from 0 to 10")


model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(RelevanceScore)


def query_llm(content, question):
    """Query the LLM to rate the relevance of a document."""
    prompt = f"""
You are tasked with evaluating the relevance of a pull request (PR) document to a given user query. 
The goal is to assess how well this PR aligns with the query in the context of building a search engine for a Retrieval-Augmented Generation (RAG) application. 
Each PR is a chunk of data stored in a text file and includes metadata, descriptions, and file changes.

Rate the relevance of the PR document to the user query on a scale from 0 to 10. Follow these specific instructions:

1. **Evaluation Criteria**:
    - Assess how specifically and accurately the PR addresses or provides relevant information for the query.
    - Consider whether the PR contains meaningful file changes, code implementations, or documentation that directly answer the query.

2. **Scoring Scale**:
    - **10**: The PR perfectly matches the query, providing complete implementation, precise details, and thorough documentation.
        Example: the PR includes the full implementation of the login page, its components, routing, and styling.
    - **9**: The PR is highly relevant, providing most of the implementation but missing minor details.
        Example: the PR includes the components and routing but lacks styling or error handling.
    - **8**: The PR is very relevant, addressing the query but with noticeable gaps in coverage or precision.
        Example: the PR includes partial implementation but lacks comprehensive testing or edge-case handling.
    - **7**: The PR is relevant but may lack significant aspects needed to fully answer the query.
        Example: the PR discusses adding authentication components but does not include the actual login page.
    - **6**: The PR is moderately relevant, providing tangential or limited information related to the query.
        Example: the PR includes code for authentication but focuses on unrelated components.
    - **5**: The PR is partially relevant but contains significant gaps or focuses on adjacent topics.
        Example: the PR references form controls but lacks any mention of a login page.
    - **4**: The PR is slightly relevant, with minimal alignment to the query.
        Example: the PR includes changes to Angular components unrelated to login functionality.
    - **3**: The PR is poorly relevant, with only marginal connection to the query.
        Example: the PR mentions Angular v2 in passing but focuses on unrelated features.
    - **2**: The PR is barely relevant, with a negligible connection to the query.
        Example: the PR is about an entirely different module but uses Angular v2.
    - **1**: The PR is irrelevant, with almost no connection to the query.
        Example: the PR is about backend features with no mention of Angular or login pages.
    - **0**: The PR is completely off-topic, providing no useful information for the query.
        Example: the PR is about a Python backend service unrelated to the query.

3. **Response Format**:
    - Your response must be a single integer from 0 to 10.
    - Do **not** include any text, explanation, or additional characters.

User Query: "{question}"
Pull Request Document:
{content}

Respond with the relevance score only (e.g., 7).
"""
    response = structured_llm.invoke(prompt)
    print(response.score)
    # try:
    #     score = int(re.search(r"\d+", response).group())
    # except (AttributeError, ValueError):
    #     score = 0  # Default score if parsing fails
    return response.score


# Define thresholds for relevance
RELEVANCE_THRESHOLD = 7

# Step 4: Evaluate each question
results = {}
overall_metrics = {
    "Precision": [],
    "MAP": [],
    "MRR": [],
    "nDCG": [],
    "average_score": [],
    "max_score": []
}

for question in questions:
    # print("Evaluating: ", question)

    # Retrieve top N documents using BM25
    retrieved_docs = retriever.retrieve(question, top_n=5)

    doc_scores = []
    for doc_index, score in retrieved_docs:
        content = documents[doc_index]  # Get the document content using index
        score = query_llm(content, question)  # Evaluate relevance with LLM
        doc_scores.append(score)

    # Normalize scores to binary relevance (1 for relevant, 0 for not relevant)
    binary_relevance = [1 if score >=
                        RELEVANCE_THRESHOLD else 0 for score in doc_scores]

    # Calculate Precision
    relevant_count = sum(binary_relevance)
    precision = relevant_count / len(doc_scores) if doc_scores else 0

    # Calculate Average Precision
    precision_values = [
        sum(binary_relevance[:i + 1]) / (i + 1)
        for i in range(len(binary_relevance))
        if binary_relevance[i] == 1
    ]
    average_precision = sum(precision_values) / \
        relevant_count if relevant_count > 0 else 0

    # Add MAP (Mean Average Precision)
    map_score = average_precision  # For a single query, MAP equals Average Precision
    overall_metrics["MAP"].append(map_score)

    # Calculate MRR
    mrr = 0
    for i, is_relevant in enumerate(binary_relevance):
        if is_relevant == 1:
            mrr = 1 / (i + 1)
            break

    # Calculate cumulative score and nDCG
    cumulative_score = sum(doc_scores)
    dcg = sum(binary_relevance[idx] / np.log2(idx + 2)
              for idx in range(len(binary_relevance)))
    idcg = sum(1 / np.log2(idx + 2) for idx in range(relevant_count))
    ndcg = dcg / idcg if idcg > 0 else 0

    average_score = np.mean(doc_scores) if doc_scores else 0

    max_score = max(doc_scores) if doc_scores else 0

    # Add metrics for this query
    results[question] = {
        "scores": doc_scores,
        "cumulative_score": cumulative_score,
        "average_score": average_score,
        "max_score": max_score,
        "Precision": precision,
        "MAP": map_score,
        "MRR": mrr,
        "nDCG": ndcg,
    }

    # Aggregate overall metrics
    overall_metrics["Precision"].append(precision)
    overall_metrics["MRR"].append(mrr)
    overall_metrics["nDCG"].append(ndcg)
    overall_metrics["average_score"].append(average_score)
    overall_metrics["max_score"].append(max_score)

# Compute overall averages
results["overall_averages"] = {
    metric: np.mean(values) for metric, values in overall_metrics.items()
}

# Save results to JSON files
with open("evaluation_results_v2.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results have been saved to evaluation_results.json.")
print("Overall metrics have been saved to overall_metrics.json.")
