import os
from dotenv import load_dotenv
import openai
import json
from statistics import mean
from pydantic import BaseModel, Field
import glob
from typing import List

# Additional imports for metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import CrossEncoder

# LangChain imports for retrieval
from bm25 import BM25Retriever, _read_documents_from_folder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate


# Uncomment if necessary
nltk.download('wordnet', quiet=True)  # Ensure wordnet is downloaded for METEOR

###################################
# Metrics and Model Initialization
###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_model.eval()


def compute_bleu(pred: str, ref: str) -> float:
    ref_tokens = ref.strip().split()
    pred_tokens = pred.strip().split()
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens,
                         smoothing_function=smoothie)
    return float(bleu)


def compute_meteor(pred: str, ref: str) -> float:
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()
    meteor = meteor_score([ref_tokens], pred_tokens)
    return float(meteor)


def compute_bertscore(pred: str, ref: str) -> float:
    P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)
    return float(F1.mean().item())


def compute_perplexity(text: str, model_id: str = 'gpt2') -> float:
    """
    Compute the perplexity of a given text using a specified pre-trained language model.

    Args:
        text (str): The input text for which to compute perplexity.
        model_id (str): The name of the pre-trained model to use (default is 'gpt2').

    Returns:
        float: The computed perplexity score.
    """
    # Load the perplexity metric
    perplexity = evaluate.load("perplexity", module_type="measurement")

    # Compute perplexity
    results = perplexity.compute(
        model_id=model_id, add_start_token=True, data=[text])

    # Return the mean perplexity
    return results['mean_perplexity']

########################
# LLM Accuracy via Relevance Score
########################


class RelevanceScore(BaseModel):
    score: int = Field(
        description="The relevance score of the prediction from 0 to 10.")


model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(RelevanceScore)


def compute_accuracy_llm(pred: str, ref: str) -> float:
    """Use an LLM to compute the accuracy score."""
    prompt = f"""
You are tasked with evaluating the accuracy of a predicted answer compared to a reference answer. 
Provide a relevance score from 0 to 10 based on the following criteria:
Matching/Accurate = the information is correct (investigation + code to be examined + proposed solutions)

### Evaluation Context:
You are assessing whether the predicted answer matches the reference answer in terms of correctness and completeness. The evaluation focuses on:
- Identifying the **correct files** to look at (based on the described investigation and implementation).
- Proposing the **correct implementation steps** aligned with the reference structure.
- Following best practices such as documentation, clear instructions, and identifying possible pitfalls.

### Scoring Criteria:
Provide a score from 0 to 10 based on the following:
- **10**: The predicted answer is completely accurate, matching the reference answer in all aspects (files, implementation, and structure).
- **9**: The predicted answer is nearly accurate but misses minor details or contains slight inaccuracies.
- **8**: The predicted answer is mostly accurate but has noticeable gaps or deviations from the reference.
- **7**: The predicted answer is moderately accurate but lacks some significant details or introduces minor inaccuracies.
- **6**: The predicted answer is partially accurate but with limited alignment to the reference.
- **5**: The predicted answer is minimally accurate and omits critical details or introduces significant inaccuracies.
- **4**: The predicted answer is very inaccurate, with minimal alignment to the reference.
- **3**: The predicted answer is poorly accurate and has only a marginal connection to the reference.
- **2**: The predicted answer is barely accurate, with negligible connection to the reference.
- **1**: The predicted answer is almost entirely inaccurate, with no meaningful alignment to the reference.
- **0**: The predicted answer is completely inaccurate or off-topic, providing no useful information.

### Response Format:
Provide a single integer (0-10). No explanation.

### Reference Answer:
{ref}

### Predicted Answer:
{pred}

Provide your accuracy score (0-10):
"""
    response = structured_llm.invoke(prompt)
    return response.score / 10.0  # Normalize to 0-1

########################
# LLM Call Function
########################


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7):
    """Generic LLM call."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return str(response.choices[0].message.content)

########################
# Evaluation Results Class
########################


class EvaluationResults:
    def __init__(self):
        # list of dicts: {q_id, accuracy, bleu, meteor, bertscore, perplexity}
        self.results = []

    def add_result(self, q_id: int, pred: str, ref: str):
        accuracy = compute_accuracy_llm(pred, ref)
        bleu = compute_bleu(pred, ref)
        meteor = compute_meteor(pred, ref)
        bert = compute_bertscore(pred, ref)
        ppl = compute_perplexity(pred)

        self.results.append({
            "q_id": q_id,
            "accuracy": accuracy,
            "bleu": bleu,
            "meteor": meteor,
            "bertscore": bert,
            "perplexity": ppl
        })

    def to_dict(self):
        # Return results as a dict
        if not self.results:
            return {"results": [], "averages": {}}

        avg_accuracy = mean(r["accuracy"] for r in self.results)
        avg_bleu = mean(r["bleu"] for r in self.results)
        avg_meteor = mean(r["meteor"] for r in self.results)
        avg_bert = mean(r["bertscore"] for r in self.results)
        avg_ppl = mean(r["perplexity"] for r in self.results)

        data = {
            "results": self.results,
            "averages": {
                "accuracy": avg_accuracy,
                "bleu": avg_bleu,
                "meteor": avg_meteor,
                "bertscore": avg_bert,
                "perplexity": avg_ppl
            }
        }
        return data

    def to_json(self):
        data = self.to_dict()
        return json.dumps(data, indent=2)


########################
# Best Retriever Workflow
########################

# Questions and references assumed to be present in QnA folder
qna_folder = "./QnA"
question_files = sorted(glob.glob(os.path.join(qna_folder, "question_*.txt")))
answer_files = sorted(glob.glob(os.path.join(qna_folder, "answer_*.txt")))

# Setup Retrieval
folder_path = "processed_docs"
documents = _read_documents_from_folder(folder_path)

# Create doc dictionary
doc_dict = {doc[0]: doc[1] for doc in documents}

bm25_retriever = BM25Retriever(documents)

VECTOR_DB_DIR = "final_all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)
vector_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})


class RelevanceScore(BaseModel):
    score: int = Field(
        description="The relevance score of the retrieval from 0 to 10")


model_refine = ChatOpenAI(model="gpt-4o-mini")
structured_llm_refine = model_refine.with_structured_output(RelevanceScore)


def expand_query(query: str) -> str:
    PROMPT_TEMPLATES = {
        "query_refinement": {
            "system": """Your job is to refine and expand any given user query related to a software codebase or feature. 
- Add relevant keywords or technologies that might be involved.
- Ensure the query is well-targeted. 
- The query will be used to retrieve PR documents.
- Your response must be exactly 10-15 words. Choose words that are unique and will likely retrieve relevant PRs.
""",
            "user": """Original user query: {query}\nPlease refine and expand this query to improve document retrieval."""
        },
    }

    messages = [
        {"role": "system",
            "content": PROMPT_TEMPLATES["query_refinement"]["system"]},
        {"role": "user", "content": PROMPT_TEMPLATES["query_refinement"]["user"].format(
            query=query)},
    ]
    expanded_response = model(messages)
    expanded_query = expanded_response.content.strip()
    return expanded_query


cross_encoder_model = CrossEncoder("BAAI/bge-reranker-base")
RELEVANCE_THRESHOLD = 7

# The prompt structure for final answer generation
system_prompt = "You are a helpful assistant that provides solutions. Follow instructions carefully."
template = """
Now generate the answer in this format. 
(Do not put “purpose” again in your response, The purpose is for your reference.)

General Answer Structure
1. Investigation / Analysis
Purpose: Understand the problem or request thoroughly before attempting a fix or implementation.
* If it is FIXING ISSUE → identify what should be tested.
* Contextual Review: Summarize the scenario or requirement. For example, "We need to add virtualization to the NextUI Table component to handle large datasets efficiently."
* Inspection of Existing Code and Behavior: Mention which files, components, or logic currently handle the functionality. For example, "Check table.tsx and related hooks that manage the Table selection logic."
* Relevant Files (to look at)
    * DIFFERENTIATE THE NEW FILES YOU SHOULD CREATE OR THE FILES YOU SHOULD TAKE A LOOK AT. IN THIS SECTION YOU ONLY WANT THE FILES YOU SHOULD SEE AND NOT CREATE
    * checkbox.tsx: Handles rendering of checkboxes.
    * table.tsx: Manages table logic, including row selection.
    * use-checkbox.ts: Defines the checkbox component’s internal logic.
* Error and Performance Analysis: If it's a bug, detail the root cause. If it's a new feature or refactor, identify current limitations. For instance, "The issue may arise from focus handling in checkboxes or the current event listeners causing multiple click registrations."
2. Proposed Solution / Implementation Steps
Purpose: Outline the solution clearly, focusing on what changes need to be made and where.
* Files to create or add
* IN THIS SECTION YOU WANT THE FILES YOU SHOULD CREATE.
    * Create new file in this folder. this file purpose is to …
* High-Level Changes: A brief overview of the strategy, e.g., "Introduce virtualization using @tanstack/react-virtual and adjust selection logic to ensure single focus event triggers."
* Specific Technical Steps:
    * Dependencies: "Install or update @tanstack/react-virtual to enable virtualization."
    * Refactoring / Adding Features: "Refactor the checkbox rendering logic in checkbox.tsx to leverage a hiddenInput slot and ensure proper focus states."
    * Updating Rendering Logic: "In table.tsx, update how rows handle selection events, ensuring that clicking a checkbox triggers a single, controlled event."
    * Improving or Adding Properties: "Add a new prop, isVirtualized, to control virtualization behavior and ensure backward compatibility."
3. Post-Fix / Post-Implementation Checks
Purpose: Validate that the solution works and does not introduce regressions.
* update unit test if necessary
* Testing and Verification: "Run unit tests and user interaction tests to ensure that rows are selectable again and that no new focus issues arise."
* Performance and Reliability: "Check if virtualization improves rendering performance on large datasets and that no extra clicks are registered."
4. Documentation and Communication (if needed)
Purpose: Ensure that all changes are clearly communicated and easy to understand for future maintainers and users.
* Update Documentation: "Add usage examples and prop definitions for the new virtualization feature in docs/components/table.mdx."
* Add Release Notes: "In the project’s CHANGELOG, mention the new virtualization support and the fixed interaction bug."
"""
PROMPT_TEMPLATES = {
    "query_refinement": {
        "system": """You are a senior developer at a large company. Your job is to refine and expand any given user query related to a software codebase or feature. The user is a newcomer developer who needs more specific, detailed, and actionable queries to find the right documents. Your refined query should:
- Add relevant keywords or technologies that might be involved.
- Ensure the query is well-targeted so that the retrieval system can find the most pertinent PRs or code references. (there is no need to mention "check for PR" as everything in the database is PR already. You should focus on the component name, concepts, tech, or terms)
Your query will be passed to search engine or vector retrieval. 
Priortize on querying the component name.
Your response must immediately begin with the query - Do not put "Relevant Query:" at the beginning.
""",
        "user": """Original user query: {query}\nPlease refine and expand this query to improve document retrieval."""
    },
    "summarization": {
        "system": """You are a senior developer summarizing a document (e.g., a PR). The summary should:
- Focus on the context relevant to the given query.
- Highlight code changes that might be helpful or good to know for the developers.
- Highlight which files or modules are touched.
- Mention coding conventions, library usage, or patterns observed.
- Note any pitfalls, best practices, or insights from PR discussions.
- Identify if there are utilities or frameworks already used, so a new developer doesn't have to re-invent solutions.
If the context does not provide relevant topics, skip them.
Your answer must be concise.
""",
        "user": """Query: {query}\nDocument Content:\n{content}\nSummarize the above document with these objectives in mind."""
    },
    "final_answer": {
        "system": """You are a senior developer synthesizing the final answer for a newcomer developer. You have several summarized documents (PRs) that relate to the user's question. Your final answer should:
- Directly answer the user's question.
- Suggest files or directories to look at.
- Highlight relevant coding patterns, naming conventions, or established best practices found in the PRs.
- Explain your findings in the previous PRs that you have explored.
- Suggest libraries or existing utilities that can be leveraged.
- Point out common pitfalls and how previous developers addressed them.
- Include why certain decisions were made and how to align with the project's standards.
If the context does not provide you with a relevant topics - skip explaining about that topics.
Your answer must be detailed but using easy to understand and concise language.
This is about empowering the new developer to ramp up quickly and make informed decisions based on relevant previous PRs.
Remember to make sure your response answer the user question as well.
""",
        "user": """User question: {query}\nContext from Summarized Pull Request Documents:\n{context}\nProvide the best possible answer, incorporating all the mentioned developer onboarding guidance."""
    }
}


def summarize_document(query, content):
    system_prompt = PROMPT_TEMPLATES["summarization"]["system"]
    user_prompt = PROMPT_TEMPLATES["summarization"]["user"].format(
        query=query, content=content)
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)


if __name__ == "__main__":
    evaluator = EvaluationResults()

    # We'll store logs for each question in a separate data structure
    logs = []

    # Load Q/A pairs
    for i, (q_file, a_file) in enumerate(zip(question_files, answer_files), start=1):
        with open(q_file, "r") as fq, open(a_file, "r") as fa:
            question = fq.read().strip()
            reference_answer = fa.read().strip()

        # Retrieval and Reranking Workflow
        expanded_query_str = expand_query(question)

        print("expanded query", expanded_query_str)

        # BM25 retrieval
        bm25_results = bm25_retriever.retrieve(expanded_query_str, top_n=10)
        bm25_docs = [(documents[idx][0], documents[idx][1])
                     for idx, _ in bm25_results]

        # Vector retrieval
        vector_retriever.search_kwargs["k"] = 10
        vector_results = vector_retriever.invoke(expanded_query_str)
        vector_docs = [(doc.metadata["file_name"], doc.page_content)
                       for doc in vector_results]

        combined_docs = {}
        for fname, content in bm25_docs:
            if fname not in combined_docs:
                combined_docs[fname] = content
        for fname, content in vector_docs:
            if fname not in combined_docs:
                combined_docs[fname] = content

        doc_items = list(combined_docs.items())  # [(fname, content), ...]
        pairs = [(expanded_query_str, content) for fname, content in doc_items]
        rerank_scores = cross_encoder_model.predict(pairs)

        scored_docs = list(zip(doc_items, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Take top 5 after reranking
        final_docs = scored_docs[:5]

        # Choose top 3 PR to use as context
        top_3_docs = final_docs[:3]

        # Summarize

        list_of_summary_string = []
        for ((fname, content), _) in top_3_docs:
            summary = summarize_document(question, content)
            list_of_summary_string.append((fname, summary))

        # Create context from top 3 docs
        context_str = "\n\n".join(
            [f"---\n{fname}\n{content}\n" for (fname, content) in list_of_summary_string])

        # Generate answer using LLM with context
        user_prompt = f"{question}\n\nContext from Top PRs:\n{context_str}\n\n{template}"
        predicted_answer = call_llm(
            system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7)

        # Evaluate result
        evaluator.add_result(i, predicted_answer, reference_answer)

        # Get the last result from evaluator to store in logs
        last_result = evaluator.results[-1]

        # Store logs
        logs.append({
            "q_id": i,
            "question": question,
            "expanded_query": expanded_query_str,
            "context_docs": [fname for ((fname, _), _) in top_3_docs],
            "context": context_str,
            "predicted_answer": predicted_answer,
            "reference_answer": reference_answer,
            "metrics": last_result
        })

    # Print out results as JSON
    evaluation_data = evaluator.to_dict()

    # Store evaluation results
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_data, f, indent=2)

    # Store logs
    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("Evaluation results stored in evaluation_results.json")
    print("Logs stored in logs.json")
