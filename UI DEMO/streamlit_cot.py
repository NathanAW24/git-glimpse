import streamlit as st
import re
import os
from dotenv import load_dotenv
from retriever import EnsembleRetriever
import openai
from concurrent.futures import ThreadPoolExecutor
import json

# Load environment variables for OpenAI API key
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
# 2. The LLM refines the query before retrieval. (Old step)
# 3. Introduce a chain-of-thought approach with up to 3 "thought cycles":
#    - Each thought cycle:
#       a. Generates new refined queries
#       b. Retrieves and summarizes docs
#       c. Produces a partial answer
#    - Accumulate partial answers and context over these 3 cycles.
# 4. The final answer is generated using the summarized content and partial answers.
# 5. Add citations afterwards.

# Initialize retriever
folder_path = "processed_docs"  # adjust as needed
vector_db_dir = "vector_db"
model_name = "thenlper/gte-small"
retriever = EnsembleRetriever(
    folder_path, vector_db_dir, model_name=model_name, device="cpu")

st.set_page_config(page_title="RAG Demo", page_icon="🌐", layout="wide")

st.title("RAG Demo: Retrieval Augmented Generation with Chain-of-Thought")

# Explanation / Instructions
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
Expand or collapse steps to keep the UI tidy.
""")

#########################################
# Prompt Templates (unchanged)
#########################################
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

#########################################
# LLM Helper Functions
#########################################


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


def prompt_clean_and_expand(query):
    system_prompt = PROMPT_TEMPLATES["query_refinement"]["system"]
    user_prompt = PROMPT_TEMPLATES["query_refinement"]["user"].format(
        query=query)
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)


def summarize_document(query, content):
    system_prompt = PROMPT_TEMPLATES["summarization"]["system"]
    user_prompt = PROMPT_TEMPLATES["summarization"]["user"].format(
        query=query, content=content)
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)


def generate_final_answer(query, summaries):
    context = "\n\n".join(
        [f"Document {i+1} summary:\n{summ}" for i, summ in enumerate(summaries)])
    system_prompt = PROMPT_TEMPLATES["final_answer"]["system"]
    user_prompt = PROMPT_TEMPLATES["final_answer"]["user"].format(
        query=query, context=context)
    return call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0)


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
        system_prompt, user_prompt, model="gpt-4o", temperature=0)
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


#########################################
# Chain-of-Thought Logic
#########################################
# We will mimic a chain-of-thought approach:
# Maximum 3 thought cycles.
# Each thought cycle:
#   - Generate new queries related to the refined question and partial answers.
#   - Retrieve documents for each query.
#   - Summarize retrieved docs.
#   - Produce partial answer based on accumulated context and previous partial answers.

# We'll define helper prompts for the chain-of-thought queries and partial answers:
COT_PROMPT_TEMPLATES = {
    "thought_process": {
        "system": """You are a senior developer using a chain-of-thought approach. You have a user question and some partial context. You will break the problem into multiple thought steps. At each step:
- Consider what additional information is needed.
- Generate 1-3 new refined queries (short, unique terms) to retrieve more docs. Focus on unique terms.
- Do NOT solve the user's question fully yet. Just propose queries to get more info.

Return the queries in a JSON array, no extra text.""",
        "user": """User question: {question}
Current partial answers: {partial_answers}
Current known context (summaries): {current_context}

Suggest 1-3 new refined queries to retrieve more information, in JSON array format:
["query1", "query2", ...]"""
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


def call_llm_raw(system, user, model="gpt-4o-mini", temperature=0):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


def get_thought_queries(question, partial_answers, current_context):
    system_prompt = COT_PROMPT_TEMPLATES["thought_process"]["system"]
    user_prompt = COT_PROMPT_TEMPLATES["thought_process"]["user"].format(
        question=question,
        partial_answers=partial_answers,
        current_context=current_context
    )
    response = call_llm_raw(system_prompt, user_prompt,
                            model="gpt-4o-mini", temperature=0)
    try:
        queries = json.loads(response)
        if not isinstance(queries, list):
            queries = []
    except:
        queries = []
    return queries


def get_partial_answer(question, all_context, partial_answers):
    system_prompt = COT_PROMPT_TEMPLATES["partial_answer"]["system"]
    user_prompt = COT_PROMPT_TEMPLATES["partial_answer"]["user"].format(
        question=question,
        all_context=all_context,
        partial_answers=partial_answers
    )
    return call_llm_raw(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7)


def retrieve_and_summarize_docs_for_queries(queries, refined_query_context):
    # For each query, retrieve and summarize top docs
    # Return list of summaries
    summaries_for_all_queries = []
    for q in queries:
        with st.spinner(f"Retrieving documents for query: {q}..."):
            retrieved_docs = retriever.get_documents(q)
        query_section = f"**Query:** {q}"
        references = []
        doc_names = list(retrieved_docs.keys())
        doc_contents = list(retrieved_docs.values())

        # Summarize these docs
        with st.write(f"Show retrieved documents for query: {q}"):
            for fname, content in retrieved_docs.items():
                pr_info = extract_pr_info(content)
                if pr_info:
                    pr_num, pr_title, pr_url = pr_info
                    st.markdown(f"- **PR #{pr_num}**: [{pr_title}]({pr_url})")
                    references.append(
                        {"pr_number": pr_num, "title": pr_title, "url": pr_url})
                else:
                    st.markdown(f"- **Document:** {fname} (No PR info found)")

        st.write("Summarizing documents for this query...")

        # Summarize in parallel
        summaries = [None]*len(doc_names)

        def summarize_task(i, query, content):
            return i, summarize_document(query, content)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(
                summarize_task, i, q, doc_contents[i]) for i in range(len(doc_names))]
            for f in futures:
                i, summary = f.result()
                summaries[i] = summary

        # Display summaries in collapsible sections
        with st.expander(f"Summaries for query: {q}", expanded=False):
            for i, fname in enumerate(doc_names):
                st.markdown(f"**Document: {fname}**")
                with st.expander("View Summary"):
                    st.write(summaries[i])
                with st.expander("View Full Content"):
                    st.write(doc_contents[i])

        # Combine all summaries for this query into one string block
        combined_summaries = "\n\n".join(
            [f"Summary of {doc_names[i]}:\n{summaries[i]}" for i in range(len(doc_names))])
        # Add to global
        summaries_for_all_queries.append(
            f"{query_section}\n{combined_summaries}")

    # Return combined context from all queries
    return "\n\n".join(summaries_for_all_queries)


#########################################
# Streamlit Workflow
#########################################
user_query = st.text_input(
    "Ask a question:", "How to add virtualization support to NextUI component?")
run_button = st.button("Run")

if run_button:
    # Step 1: Query Refinement
    st.markdown("### 1️⃣ Step 1: Query Refinement")
    refined_query = prompt_clean_and_expand(user_query)
    st.write("**Refined Query:**", refined_query)

    # We now implement the chain-of-thought approach
    # We'll do a maximum of 3 thoughts
    max_thoughts = 3
    partial_answers = []
    all_context_pieces = []

    for thought_i in range(1, max_thoughts+1):
        st.markdown(f"### Thought Cycle {thought_i}")
        with st.expander(f"Initiate Thought {thought_i}", expanded=True):
            current_context = "\n\n".join(all_context_pieces)
            current_partial_answers = "\n\n".join(partial_answers)

            # Get queries for this thought
            queries = get_thought_queries(
                user_query, current_partial_answers, current_context)
            if not queries:
                st.write("No new queries generated. Stopping chain-of-thought.")
                break

            st.write("**Queries generated:**", queries)
            # Retrieve and summarize docs for these queries
            context_from_queries = retrieve_and_summarize_docs_for_queries(
                queries, refined_query)
            # Add the new context
            all_context_pieces.append(context_from_queries)

            # Produce a new partial answer
            st.write(
                "Generating partial answer based on current aggregated context and previous partial answers...")
            combined_context = "\n\n".join(all_context_pieces)
            current_partial_answers = "\n\n".join(partial_answers)
            new_partial_answer = get_partial_answer(
                user_query, combined_context, current_partial_answers)
            partial_answers.append(new_partial_answer)

            with st.expander(f"Partial Answer after Thought {thought_i}", expanded=False):
                st.write(new_partial_answer)

    # After all thoughts, produce final answer
    st.markdown("### Final Answer Generation")
    st.write(
        "Combining all partial answers and context to produce the final answer...")
    combined_context = "\n\n".join(all_context_pieces)
    # Let's just re-retrieve top docs for the final refined query or just use the context we have
    # We already have summaries in combined_context. We'll treat all_context_pieces as summarized docs.

    # We do final answer generation
    final_answer = generate_final_answer(user_query, all_context_pieces)

    st.write("### 🗒️ **Final Answer (Before Citation):**")
    st.write(final_answer)

    # Extract references from all contexts
    # We have references when we retrieved docs. Let's do a quick extraction by searching PRs in all_context_pieces:
    # Actually references were only displayed in UI, we did not store them. Let's do a naive approach:
    # The final annotated citations step might fail if no references recognized, but let's try:
    # We'll just re-run retrieval on final refined query and extract references for citations:
    retrieved_docs_final = retriever.get_documents(refined_query)
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
