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


# Load environment variables
load_dotenv()

# Set OpenAI API Key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

# Step 1: Define your questions for evaluation
questions = [[
    "pr_data_0_doc_7.txt",
    "How can I enhance the validation capabilities of a Select component to prevent the default browser error UI from appearing, while also supporting required fields, custom validation functions, and server-side validation? Additionally, what changes would be needed to integrate these features into the multi-select component for better user interaction and validation logic?"
],
    [
        "pr_data_21_doc_3.txt",
        "How can I extend the props for the NextUIProvider to allow direct styling of the container element created by the OverlayProvider, and ensure that the children components are correctly inheriting styles from the body element? Additionally, how can I update the default locale setting within the NextUIProvider without introducing breaking changes?"
],
    [
        "pr_data_27_doc_4.txt",
        "What is the process for fixing documentation typos in our project? Are there any specific guidelines or steps I should follow when making minor text corrections, such as in the Table documentation?"
],
    [
        "pr_data_3_doc_7.txt",
        "How should I refactor the documentation for the `Kbd` component to improve the developer experience, specifically when demonstrating key combinations? Are there any existing patterns or examples in the codebase that show how to separate JSX code into raw files for better documentation organization?"
],
    [
        "pr_data_2_doc_39.txt",
        "How can I refactor the documentation examples for the Tabs component to improve developer experience by converting inline code examples into separate raw JSX files and ensuring they are consistent across different use cases like colors, sizes, and variants?"
],
    [
        "pr_data_2_doc_1.txt",
        "How can I resolve a ReferenceError related to image handling in Server-Side Rendering (SSR) contexts? Specifically, what changes are necessary in the `useImage` hook to improve image loading logic and ensure proper handling of image references during SSR?"
],
    [
        "pr_data_12_doc_33.txt",
        "How can I fix the issue where the `onPress` event handler is not being triggered in the `DropdownItem` component? Specifically, what changes are needed to ensure that all event handlers, such as `onPress`, `onPressStart`, `onPressEnd`, `onPressChange`, and `onPressUp`, work correctly in the NextUI library while maintaining efficient event handling within the dropdown menu components?"
],
    [
        "pr_data_7_doc_9.txt",
        "How should I approach fixing the issue with null `defaultVariants` in `extendVariants` for the system-rsc component? Additionally, are there any specific considerations for adjusting the calendar content's width to align with visible months in the date-range-picker, and how can I ensure consistent lowercase normalization for the 'status' field across multiple files?"
],
    [
        "pr_data_4_doc_31.txt",
        "How can I modify the ListBox and Select components to support numeric keys without converting them to strings, while ensuring that the key property remains optional? Additionally, what changes are necessary to update the tests to accommodate these modifications and maintain proper selection functionality?"
],
    [
        "pr_data_3_doc_24.txt",
        "How can I refactor our documentation examples to separate raw JSX/TSX code from their respective TypeScript files for better maintainability, while ensuring that the Calendar component in our documentation demonstrates various functionalities such as controlled focus, disabled states, international calendars, and date presets? Additionally, how should I handle breaking changes, if any, during this refactor to ensure backward compatibility?"
],    [
        "pr_data_29_doc_45.txt",
        "How can I make a `Pagination` component properly controlled by syncing its internal state with an external `page` prop?"
],
    [
        "pr_data_15_doc_4.txt",
        "How do I set up and configure CodeRabbit for automatic code reviews, language preferences, and chat auto-replies in our project?"
],
    [
        "pr_data_12_doc_18.txt",
        "How can I fix the issue where an input field with the isClearable option fails to clear its value and ensure the onClear callback is triggered correctly, possibly using react-hook-form for enhanced form handling?"
],
    [
        "pr_data_14_doc_12.txt",
        "How do I fix the issue with refs in lazy-loaded motion components using React.forwardRef()?"
],
    [
        "pr_data_23_doc_9.txt",
        "How to prevent a tooltip from reopening if the cursor moves out before the delay is reached?"
],
    [
        "pr_data_2_doc_2.txt",
        "How should the `buildLocation` function be correctly used with objects in `NextUIProvider` for routing?"
],
    [
        "pr_data_6_doc_39.txt",
        "How can I clean up unused and redundant code in our Storybook setup without introducing breaking changes?"
],
    [
        "pr_data_5_doc_48.txt",
        "How can I ensure that the date validation error messages in the date-picker reflect the locale specified in NextUIProvider rather than the browser's default locale?"
],
    [
        "pr_data_10_doc_22.txt",
        "How do I update the documentation and marketing components to announce a new version release while incorporating global animation settings, API improvements, and CLI enhancements?"
],
    [
        "pr_data_20_doc_29.txt",
        "How can I remove the extra bottom space in the Select component when using helper components to ensure consistent layout with the Input component?"
]]

# Step 2: Load vector database
VECTOR_DB_DIR = "vector_db"
MODEL_NAME = "thenlper/gte-small"
# Use OpenAI embeddings if applicable
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    # Ensure the device matches the previous setup
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})

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

for [file_name_query, question] in questions:
    print("Evaluating: ", question)

    # Retrieve top N documents using BM25
    retrieved_docs = retriever.invoke(question)
    doc_scores = []
    for doc in retrieved_docs:
        # Get the document content using index
        print(doc.metadata)
        file_name, content = doc.metadata["file_name"], doc.page_content
        if file_name == file_name_query:
            doc_scores.append(10)
            print("Match found!", file_name)
        else:
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
