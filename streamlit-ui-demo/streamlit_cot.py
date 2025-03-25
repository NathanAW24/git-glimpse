import evaluate
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_chroma import Chroma
from bm25 import BM25Retriever, _read_documents_from_folder
from sentence_transformers import CrossEncoder
import numpy as np
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from typing import List
import glob
from statistics import mean
import json
from concurrent.futures import ThreadPoolExecutor
import openai
from retriever import EnsembleRetriever
from dotenv import load_dotenv
import os
import re
import streamlit as st
import sys

# LangChain imports for retrieval

nltk.download('wordnet', quiet=True)  # Ensure wordnet is downloaded for METEOR

#########################################
# Setup and Load Keys
#########################################
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Explanation of GitGlimpse (for context, not part of the app logic):
# GitGlimpse is a specialized search engine designed to help developers quickly
# locate and understand relevant pull requests (PRs) from vast repositories.
# By leveraging Retrieval-Augmented Generation (RAG) and LLM-powered summarization,
# GitGlimpse transforms PR data into actionable insights. Developers can input
# queries about specific tasks or features, and GitGlimpse retrieves and ranks PRs
# based on relevance, providing concise summaries and context. This reduces time
# spent searching through unrelated PRs, accelerating feature development and
# enhancing collaboration.

# Project Goals:
# 1. User enters a question.
# 2. The LLM refines the query before retrieval.
# 3. Introduce a chain-of-thought approach with up to 3 "thought cycles":
#    - Each thought cycle:
#       a. Generates new refined queries
#       b. Retrieves and summarizes docs
#       c. Produces a partial answer
#    - Accumulate partial answers and context over these 3 cycles.
# 4. The final answer is generated using the summarized content and partial answers.
# 5. Add citations afterwards.
# Additionally, we incorporate evaluation metrics and final result logging from the previously shown code snippet.

#########################################
# Model and Metric Initialization
#########################################
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


model_eval = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model_eval.with_structured_output(RelevanceScore)


def compute_accuracy_llm(pred: str, ref: str) -> float:
    prompt = f"""
You are tasked with evaluating the accuracy of a predicted answer compared to a reference answer. 
Provide a relevance score from 0 to 10 based on the following criteria:
Matching/Accurate = the information is correct (investigation + code to be examined + proposed solutions)

### Evaluation Context:
You are assessing whether the predicted answer matches the reference answer in terms of correctness and completeness. The evaluation focuses on:
- Identifying the **correct files** to look at
- Proposing the **correct implementation steps**
- Following best practices, documentation, clear instructions, and identifying possible pitfalls.

### Scoring Criteria:
(0 to 10, as described previously)

### Reference Answer:
{ref}

### Predicted Answer:
{pred}

Provide your accuracy score (0-10):
"""
    response = structured_llm.invoke(prompt)
    return response.score / 10.0


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


#########################################
# Retrieval Setup
#########################################
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

model = ChatOpenAI(model="gpt-4o-mini")
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


def summarize_document_func(query, content):
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


def extract_pr_info(text):
    pr_number_pattern = r"Pull Request Number:\s*(\d+)"
    pr_title_pattern = r"Title:\s*(.*)"
    pr_url_pattern = r"URL:\s*(https?://[^\s]+)"

    pr_num = re.search(pr_number_pattern, text)
    pr_title = re.search(pr_title_pattern, text)
    pr_url = re.search(pr_url_pattern, text)

    if pr_num and pr_url:
        return pr_num.group(1), (pr_title.group(1) if pr_title else "No Title Found"), pr_url.group(1)
    return None


def add_citations_to_answer(final_answer, references):
    refs_str = "\n".join(
        [f"PR Number: {ref['pr_number']}, Title: {ref['title']}, URL: {ref['url']}" for ref in references]
    )

    system_prompt = """You are a post-processing assistant that adds citations to a previously generated answer.
You are given a final answer text and a list of PR references. Insert citations in the answer where relevant.
Even if it is slightly relevant or little relevant, YOU SHOULD STILL PUT THE CITATION.
Ideally every line of your answer or every information should be cited by citation.

Rules:
- Use the format: <citation pr_number="..." title="..." url="..." />.
- Do not remove or alter existing text, only add citations.
- Ensure the citations are parseable by regex.

"""
    user_prompt = f"""
Final Answer Text:
{final_answer}

References (PRs):
{refs_str}

Please insert citations as instructed.
"""
    annotated_answer = call_llm(
        system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)
    return annotated_answer


def parse_citations(annotated_answer):
    pattern = r'<citation\s+pr_number="([^"]+)"\s+title="([^"]+)"\s+url="([^"]+)"\s*/>'
    matches = re.findall(pattern, annotated_answer)
    citations = []
    for pr_number, title, url in matches:
        citations.append({"pr_number": pr_number, "title": title, "url": url})
    return citations


def replace_citations_with_links(text):
    pattern = r'<citation\s+pr_number="([^"]+)"\s+title="([^"]+)"\s+url="([^"]+)"\s*/>'

    def repl(match):
        pr_number = match.group(1)
        title = match.group(2)
        url = match.group(3)
        return f"[(PR #{pr_number})]({url})"
    return re.sub(pattern, repl, text)


template = """
Now generate the answer in this format. 
(Do not put “purpose” again in your response, The purpose is for your reference.)

General Answer Structure
1. Investigation / Analysis
Purpose: Understand the problem or request thoroughly before attempting a fix or implementation.
* If it is FIXING ISSUE → identify what should be tested.
* Contextual Review: Summarize the scenario or requirement. For example, "We need to add virtualization to the NextUI Table component to handle large datasets efficiently."
* Inspection of Existing Code and Behavior: Mention which files, components, or logic currently handle the functionality.
* Relevant Files (to look at)
    * DIFFERENTIATE THE NEW FILES YOU SHOULD CREATE OR THE FILES YOU SHOULD TAKE A LOOK AT. IN THIS SECTION YOU ONLY WANT THE FILES YOU SHOULD SEE AND NOT CREATE
* Error and Performance Analysis: If it's a bug, detail the root cause.
2. Proposed Solution / Implementation Steps
Purpose: Outline the solution clearly.
* Files to create (only in this section)
* High-Level Changes
* Specific Technical Steps
3. Post-Fix / Post-Implementation Checks
Purpose: Validate the solution.
4. Documentation and Communication (if needed)
"""

system_prompt_template = "You will be given a template - follow the template. You will be given partial answers - combine them. You will be given the question and the PR context."


#########################################
# Streamlit UI
#########################################
st.set_page_config(page_title="RAG Demo", page_icon="🌐", layout="wide")
st.title("RAG Demo: Retrieval Augmented Generation with Chain-of-Thought")

st.markdown("""
### Instructions
1. Type your question into the input box below.
2. The system will:
   - Clean and expand your query using an LLM (initial step).
   - Then enter a **3-step Chain-of-Thought** process:
     - **Thought 1**: Generate queries, retrieve & summarize docs, produce a partial answer.
     - **Thought 2**: Based on current context & partial answer, generate more queries, get more docs, refine partial answer.
     - **Thought 3**: Repeat process, further refining partial answer.
   - Finally, produce a final comprehensive answer and add citations.
   
You will see the chain-of-thought steps and partial answers as we go.
""")

user_query = st.text_input(
    "Ask a question:", "How to add virtualization support to NextUI component?")
run_button = st.button("Run")

if run_button:
    # Step 1: Query Refinement
    st.markdown("### 1️⃣ Step 1: Query Refinement")
    refined_query = expand_query(user_query)
    st.write("**Refined Query:**", refined_query)

    # Chain-of-thought variables
    max_thoughts = 3
    partial_answers_list = []
    all_context_pieces = []

    for thought_i in range(1, max_thoughts+1):
        st.markdown(f"### Thought Cycle {thought_i}")
        st.write(f"Initiate Thought {thought_i}")
        current_context = "\n\n".join(all_context_pieces)
        current_partial_answers = "\n\n".join(partial_answers_list)

        # Get queries for this thought
        queries = get_thought_queries(
            user_query, current_partial_answers, current_context)
        if not queries:
            st.write("No new queries generated. Stopping chain-of-thought.")
            break

        st.write("**Queries generated:**", queries)

        # For each query, do retrieval and summarize
        def summarize_retrieved_docs(q):
            with st.spinner(f"Retrieving documents for query: {q}..."):
                retrieved_docs = EnsembleRetriever(
                    folder_path, VECTOR_DB_DIR, model_name=MODEL_NAME, device="cpu").get_documents(q)

            # Show retrieved docs
            with st.expander(f"Show retrieved documents for query: {q}", expanded=False):
                references_local = []
                for fname, content in retrieved_docs.items():
                    pr_info = extract_pr_info(content)
                    if pr_info:
                        pr_num, pr_title, pr_url = pr_info
                        st.markdown(
                            f"- **PR #{pr_num}**: [{pr_title}]({pr_url})")
                        references_local.append(
                            {"pr_number": pr_num, "title": pr_title, "url": pr_url})
                    else:
                        st.markdown(
                            f"- **Document:** {fname} (No PR info found)")

            st.write("Summarizing documents for this query...")
            doc_names = list(retrieved_docs.keys())
            doc_contents = list(retrieved_docs.values())

            # Summarize in parallel
            def summarize_task(i, q, content):
                return i, summarize_document_func(q, content)

            summaries = [None]*len(doc_names)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(
                    summarize_task, i, q, doc_contents[i]) for i in range(len(doc_names))]
                for f in futures:
                    i, summary = f.result()
                    summaries[i] = summary

            # Display summaries
            st.markdown(f"**SUMMARIES FOR QUERY: {q}**")
            for i, fname in enumerate(doc_names):
                st.markdown(f"**Document: {fname}**")
                with st.expander("View Summary"):
                    st.write(summaries[i])
                with st.expander("View Full Content"):
                    st.write(doc_contents[i])

            combined_summaries = "\n\n".join(
                [f"Summary of {doc_names[i]}:\n{summaries[i]}" for i in range(len(doc_names))])
            return f"**Query:** {q}\n{combined_summaries}"

        context_from_queries = []
        for q in queries:
            context_block = summarize_retrieved_docs(q)
            context_from_queries.append(context_block)

        # Add the new context
        query_context_combined = "\n\n".join(context_from_queries)
        all_context_pieces.append(query_context_combined)

        # Produce a new partial answer
        st.write(
            "Generating partial answer based on current aggregated context and previous partial answers...")
        combined_context = "\n\n".join(all_context_pieces)
        current_partial_answers = "\n\n".join(partial_answers_list)
        new_partial_answer = get_partial_answer(
            user_query, combined_context, current_partial_answers)
        partial_answers_list.append(new_partial_answer)

        with st.expander(f"Partial Answer after Thought {thought_i}", expanded=False):
            st.write(new_partial_answer)

    # After all thoughts, produce final answer
    st.markdown("### Final Answer Generation")
    st.write(
        "Combining all partial answers and context to produce the final answer...")
    combined_context = "\n\n".join(all_context_pieces)

    # Produce final answer using the given template
    final_user_prompt = f"{user_query}{chr(92)}n{chr(92)}nPrevious partial answers:{'{chr(92)}n{chr(92)}n'.join(partial_answers_list)}Context from Top PRs:{chr(92)}n{combined_context}{chr(92)}n{chr(92)}nWhen answering you must follow this template: {template}"
    final_answer = call_llm(
        system_prompt_template, final_user_prompt, model="gpt-4o", temperature=0.7)

    st.write("### 🗒️ **Final Answer (Before Citation):**")
    st.write(final_answer)

    # Extract references from final retrieval
    retrieved_docs_final = EnsembleRetriever(
        folder_path, VECTOR_DB_DIR, model_name=MODEL_NAME, device="cpu").get_documents(refined_query)
    references = []
    for fname, content in retrieved_docs_final.items():
        pr_info = extract_pr_info(content)
        if pr_info:
            pr_num, pr_title, pr_url = pr_info
            references.append(
                {"pr_number": pr_num, "title": pr_title, "url": pr_url})

    st.write("Adding citations to the final answer...")
    annotated_answer = add_citations_to_answer(final_answer, references)
    annotated_answer_with_links = replace_citations_with_links(
        annotated_answer)

    st.markdown("## 📝 **Final Answer (Annotated with Citations):**")
    st.write(annotated_answer_with_links)

    citations = parse_citations(annotated_answer)
    if citations:
        st.markdown("### References Found in Answer:")
        seen_citations = set()
        for c in citations:
            citation_key = (c['pr_number'], c['title'], c['url'])
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                st.markdown(
                    f"- PR #{c['pr_number']}: [{c['title']}]({c['url']})")
    else:
        st.write("No citations found.")

    # Optional: Evaluate final answer against a known reference if files exist
    # We'll do a batch evaluation as shown in original code
    # This section is only run if QnA files are available and we want to do evaluation
    evaluator = EvaluationResults()
    logs = []

    for i, (q_file, a_file) in enumerate(zip(question_files, answer_files), start=1):
        with open(q_file, "r") as fq, open(a_file, "r") as fa:
            question_data = fq.read().strip()
            reference_answer = fa.read().strip()

        # Just re-run the pipeline or skip if you don't want a separate evaluation
        # For demonstration, let's just evaluate the final_answer we produced (if question_data == user_query)
        if question_data == user_query:
            evaluator.add_result(i, final_answer, reference_answer)
            logs.append({
                "q_id": i,
                "question": question_data,
                "final_answer": final_answer,
                "reference_answer": reference_answer,
                "metrics": evaluator.results[-1]
            })

    evaluation_data = evaluator.to_dict()
    with open("evaluation_results_chain_of_thought.json", "w") as f:
        json.dump(evaluation_data, f, indent=2)

    with open("logs_chain_of_thought.json", "w") as f:
        json.dump(logs, f, indent=2)

    st.write("Evaluation results stored in evaluation_results_chain_of_thought.json")
    st.write("Logs stored in logs_chain_of_thought.json")
