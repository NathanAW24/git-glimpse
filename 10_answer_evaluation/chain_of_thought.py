import os
from dotenv import load_dotenv
import openai
import json
from statistics import mean
from pydantic import BaseModel, Field
import glob
from typing import List
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
import evaluate

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
    meteor_val = meteor_score([ref_tokens], pred_tokens)
    return float(meteor_val)


def compute_bertscore(pred: str, ref: str) -> float:
    P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)
    return float(F1.mean().item())


def compute_perplexity(text: str, model_id: str = 'gpt2') -> float:
    perplexity = evaluate.load("perplexity", module_type="measurement")
    results = perplexity.compute(
        model_id=model_id, add_start_token=True, data=[text])
    return results['mean_perplexity']


class RelevanceScore(BaseModel):
    score: int = Field(
        description="The relevance score of the prediction from 0 to 10.")


model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(RelevanceScore)


def compute_accuracy_llm(pred: str, ref: str) -> float:
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
    return response.score / 10.0


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return str(response.choices[0].message.content)


class EvaluationResults:
    def __init__(self):
        self.results = []

    def add_result(self, q_id: int, pred: str, ref: str):
        accuracy = compute_accuracy_llm(pred, ref)
        bleu = compute_bleu(pred, ref)
        meteor_val = compute_meteor(pred, ref)
        bert = compute_bertscore(pred, ref)
        ppl = compute_perplexity(pred)
        self.results.append({
            "q_id": q_id,
            "accuracy": accuracy,
            "bleu": bleu,
            "meteor": meteor_val,
            "bertscore": bert,
            "perplexity": ppl
        })

    def to_dict(self):
        if not self.results:
            return {"results": [], "averages": {}}
        avg_accuracy = mean(r["accuracy"] for r in self.results)
        avg_bleu = mean(r["bleu"] for r in self.results)
        avg_meteor = mean(r["meteor"] for r in self.results)
        avg_bert = mean(r["bertscore"] for r in self.results)
        avg_ppl = mean(r["perplexity"] for r in self.results)
        return {
            "results": self.results,
            "averages": {
                "accuracy": avg_accuracy,
                "bleu": avg_bleu,
                "meteor": avg_meteor,
                "bertscore": avg_bert,
                "perplexity": avg_ppl
            }
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


# Setup retrieval
qna_folder = "./QnA"
question_files = sorted(glob.glob(os.path.join(qna_folder, "question_*.txt")))
answer_files = sorted(glob.glob(os.path.join(qna_folder, "answer_*.txt")))

folder_path = "processed_docs"
documents = _read_documents_from_folder(folder_path)
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
    score: int = Field(description="The relevance score of retrieval")


model_refine = ChatOpenAI(model="gpt-4o-mini")

PROMPT_TEMPLATES = {
    "query_refinement": {
        "system": """You are a senior developer. Refine and expand the given user query related to a software codebase or feature:
- Add relevant keywords or technologies
- Ensure the query is well-targeted
Your response must begin immediately with the query (10-15 words)""",
        "user": """Original user query: {query}\nPlease refine and expand this query to improve document retrieval."""
    },
    "thought_process": {
        "system": """You are a senior developer using a chain-of-thought approach. You have a user question and some partial context. You will break the problem into multiple thought steps. At each step:
- Consider what additional information is needed.
- Generate 1-3 new refined queries (short, unique terms) to retrieve more docs. Focus on unique terms.
- Do NOT solve the user's question fully yet. Just propose queries to get more info.

Return the queries in a JSON array, no extra text.""",
        "user": """User question: {question}
Current partial answer: {partial_answer}
Current known context (summaries): {current_context}

Suggest 1-3 new refined queries to retrieve more information, in JSON array format:
["query1", "query2", ...]"""
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
        "system": """You are a senior developer. Combine all retrieved info and partial answers to finalize the answer for the user. Be thorough and helpful.""",
        "user": """User question: {question}
All gathered context:
{all_context}

Partial answers from previous steps:
{partial_answers}

Provide the best final answer to the user question now."""
    },
    "partial_answer": {
        "system": """You are a senior developer. Given the current aggregated context and previous partial answers, produce a new refined answer.
Your answer must be related to the context that have already been given. Your task is to refine the previous answer to include more context so that it will answer the question.
The target answer will include: how to investigate and explore the question, proposed solution, implementation steps, post-fix implementation/test, and documentation.
If the context does not provide you with the topics, you can skip expanding that section of the answer.

        """,
        "user": """User question: {question}
Current aggregated context:
{all_context}

Previous partial answers:
{partial_answers}

Produce an updated partial answer (not final)."""
    }
}

cross_encoder_model = CrossEncoder("BAAI/bge-reranker-base")
RELEVANCE_THRESHOLD = 7


def summarize_document(query, content):
    system_prompt = PROMPT_TEMPLATES["summarization"]["system"]
    user_prompt = PROMPT_TEMPLATES["summarization"]["user"].format(
        query=query, content=content)
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)


def expand_query(query: str) -> str:
    messages = [
        {"role": "system",
            "content": PROMPT_TEMPLATES["query_refinement"]["system"]},
        {"role": "user", "content": PROMPT_TEMPLATES["query_refinement"]["user"].format(
            query=query)},
    ]
    expanded_response = model(messages)
    return expanded_response.content.strip()


def retrieve_and_rerank(query_str: str):
    # Best retrieval method
    # BM25
    bm25_results = bm25_retriever.retrieve(query_str, top_n=10)
    bm25_docs = [(documents[idx][0], documents[idx][1])
                 for idx, _ in bm25_results]

    # Vector retrieval
    vector_retriever.search_kwargs["k"] = 10
    vector_results = vector_retriever.invoke(query_str)
    vector_docs = [(doc.metadata["file_name"], doc.page_content)
                   for doc in vector_results]

    combined_docs = {}
    for fname, content in bm25_docs:
        if fname not in combined_docs:
            combined_docs[fname] = content
    for fname, content in vector_docs:
        if fname not in combined_docs:
            combined_docs[fname] = content

    doc_items = list(combined_docs.items())
    pairs = [(query_str, c) for _, c in doc_items]
    rerank_scores = cross_encoder_model.predict(pairs)
    scored_docs = list(zip(doc_items, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    final_docs = scored_docs[:5]
    top_3_docs = final_docs[:3]
    return top_3_docs


def get_thought_queries(question, partial_answer, current_context):
    # Ask LLM what queries to do next
    system_prompt = PROMPT_TEMPLATES["thought_process"]["system"]
    user_prompt = PROMPT_TEMPLATES["thought_process"]["user"].format(
        question=question,
        partial_answer=partial_answer,
        current_context=current_context
    )
    response = call_llm(system_prompt, user_prompt,
                        model="gpt-4o-mini", temperature=0)
    # response should be a JSON array of queries
    try:
        queries = json.loads(response)
        if not isinstance(queries, list):
            queries = []
    except:
        queries = []
    return queries


def get_partial_answer(question, all_context, partial_answers):
    system_prompt = PROMPT_TEMPLATES["partial_answer"]["system"]
    user_prompt = PROMPT_TEMPLATES["partial_answer"]["user"].format(
        question=question,
        all_context=all_context,
        partial_answers=partial_answers
    )
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7)


def get_final_answer(question, all_context, partial_answers):
    system_prompt = PROMPT_TEMPLATES["final_answer"]["system"]
    user_prompt = PROMPT_TEMPLATES["final_answer"]["user"].format(
        question=question,
        all_context=all_context,
        partial_answers=partial_answers
    )
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0.7)


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
system_prompt = "You will be given a template to answer with guide or instruction to help onboard developer faster. You will be given a template - follow the template. You will be given partial answers - combine them. You will be given the question and the PR context. Pay attention to those"

if __name__ == "__main__":
    evaluator = EvaluationResults()
    logs = []

    for i, (q_file, a_file) in enumerate(zip(question_files, answer_files), start=1):
        with open(q_file, "r") as fq, open(a_file, "r") as fa:
            question = fq.read().strip()
            reference_answer = fa.read().strip()

        # First expand query to get a good starting point
        expanded_query_str = expand_query(question)

        # Initialize CoT variables
        all_context_summaries = []
        partial_answers = []
        current_context_str = ""

        # We do up to 3 thought cycles
        for thought_idx in range(3):
            # Generate queries to retrieve more info based on current partial answer and context
            queries = get_thought_queries(question, "\n\n".join(
                partial_answers), "\n\n".join(all_context_summaries))
            if not queries:
                # If no queries returned, break early
                break

            # For each query, do retrieval and summarization
            iteration_context = []
            for q in queries:
                # Possibly also do query expansion again to improve q
                refined_q = expand_query(q)
                top_3_docs = retrieve_and_rerank(refined_q)
                # Summarize docs
                for ((fname, content), _) in top_3_docs:
                    summary = summarize_document(refined_q, content)
                    iteration_context.append(
                        f"Query: {q}\nFile: {fname}\nSummary: {summary}")

            # Add iteration context to global context
            all_context_summaries.extend(iteration_context)

            # Produce a new partial answer after this iteration
            partial_answer = get_partial_answer(question, "\n\n".join(
                all_context_summaries), "\n\n".join(partial_answers))
            partial_answers.append(partial_answer)

        all_context_summaries = list(set(all_context_summaries))
        #
        combined_partial_answers = "\n\n".join(partial_answers)
        user_prompt = f"{question}\n\nPrevious partial answers:{combined_partial_answers}Context from Top PRs:\n{all_context_summaries}\n\nWhen answering you must follow this template (Do not put “purpose” again in your response, The purpose is for your reference.): {template}"
        final_answer = call_llm(
            system_prompt, user_prompt, model="gpt-4o", temperature=0.7)
        # After all iterations, produce final answer
        # final_answer = get_final_answer(question, "\n\n".join(
        #     all_context_summaries), "\n\n".join(partial_answers))

        # Evaluate final answer
        evaluator.add_result(i, final_answer, reference_answer)

        # Log everything
        logs.append({
            "q_id": i,
            "question": question,
            "expanded_initial_query": expanded_query_str,
            "all_context_summaries": all_context_summaries,
            "partial_answers": partial_answers,
            "final_answer": final_answer,
            "reference_answer": reference_answer,
            "metrics": evaluator.results[-1]
        })

    evaluation_data = evaluator.to_dict()
    with open("evaluation_results_chain_of_thought.json", "w") as f:
        json.dump(evaluation_data, f, indent=2)

    with open("logs_chain_of_thought.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("Evaluation results stored in evaluation_results.json")
    print("Logs stored in logs.json")
