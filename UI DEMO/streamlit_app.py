import streamlit as st
import re
import os
from dotenv import load_dotenv
from retriever import EnsembleRetriever
import openai
from concurrent.futures import ThreadPoolExecutor

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
# 2. The LLM refines the query before retrieval.
# 3. Relevant documents (PRs) are retrieved.
# 4. Each document is summarized (now done in parallel via threading).
# 5. The final answer is generated using the summarized content.
#
# The final answer guides a newcomer developer by referencing codebase structure,
# coding conventions, libraries, and best practices gleaned from PRs.

# Initialize retriever
folder_path = "processed_docs"  # adjust as needed
vector_db_dir = "vector_db"
model_name = "thenlper/gte-small"
retriever = EnsembleRetriever(
    folder_path, vector_db_dir, model_name=model_name, device="cpu")

st.set_page_config(page_title="RAG Demo", page_icon="🌐", layout="wide")

st.title("RAG Demo: Retrieval Augmented Generation")

# Explanation / Instructions
st.markdown("""
### Instructions
1. Type your question into the input box below.
2. The system will:
   - Clean and expand your query using an LLM.
   - Retrieve relevant documents from an ensemble of BM25 and vector similarity.
   - Summarize each relevant document to reduce length in parallel
   - Use the summarized content to generate a final answer that helps you onboard quickly.
   
You will see placeholders as the summaries are being processed. Once all are done, the final answer step will proceed.
""")

# Prompt Templates (unchanged)
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


def call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0.7):
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


def prompt_clean_and_expand(query):
    system_prompt = PROMPT_TEMPLATES["query_refinement"]["system"]
    user_prompt = PROMPT_TEMPLATES["query_refinement"]["user"].format(
        query=query)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


def summarize_document(query, content):
    system_prompt = PROMPT_TEMPLATES["summarization"]["system"]
    user_prompt = PROMPT_TEMPLATES["summarization"]["user"].format(
        query=query, content=content)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


def generate_final_answer(query, summaries):
    context = "\n\n".join(
        [f"Document {i+1} summary:\n{summ}" for i, summ in enumerate(summaries)])
    system_prompt = PROMPT_TEMPLATES["final_answer"]["system"]
    user_prompt = PROMPT_TEMPLATES["final_answer"]["user"].format(
        query=query, context=context)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


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


# Streamlit Workflow
user_query = st.text_input(
    "Ask a question:", "How to add virtualization support to NextUI component?")
# I want to solve flickering issue in next ui button. How can i investigate this? describe a good investigation workflow.
run_button = st.button("Run")

if run_button:
    st.subheader("Chain of Thought")

    # Step 1: Query Refinement
    st.markdown("**1️⃣ Step 1: Query Refinement**")
    refined_query = prompt_clean_and_expand(user_query)
    st.write("**Refined Query:**", refined_query)

    # Step 2: Retrieve Relevant Documents
    st.markdown("**2️⃣ Step 2: Retrieve Documents**")
    retrieved_docs = retriever.get_documents(refined_query)

    references = []
    if retrieved_docs:
        for fname, content in retrieved_docs.items():
            pr_info = extract_pr_info(content)
            if pr_info:
                pr_num, pr_title, pr_url = pr_info
                st.markdown(f"- **PR #{pr_num}**: [{pr_title}]({pr_url})")
                references.append(
                    {"pr_number": pr_num, "title": pr_title, "url": pr_url})
            else:
                st.markdown(f"- **Document:** {fname} (No PR info found)")
    else:
        st.write("No documents found.")

    # Step 3: Summarize Documents (in parallel)
    st.markdown("**3️⃣ Step 3: Summarize Documents**")

    doc_names = list(retrieved_docs.keys())
    doc_contents = list(retrieved_docs.values())

    # Create placeholders for each doc summary
    summary_placeholders = [st.empty() for _ in doc_names]

    # Initially show loading message
    for i, fname in enumerate(doc_names):
        with summary_placeholders[i].container():
            st.markdown(f"**Document: {fname}**")
            st.write("Loading summary...")

    # We'll store the final summaries here
    summaries = [None] * len(doc_names)

    def summarize_task(i, query, content):
        # Worker function for threading
        result = summarize_document(query, content)
        return i, result

    # Use ThreadPoolExecutor to run summaries in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(
            summarize_task, i, refined_query, doc_contents[i]) for i in range(len(doc_names))]

        # As each future completes, update the UI
        for f in futures:
            i, summary = f.result()
            summaries[i] = summary
            # Update the placeholder with final summary content
            summary_placeholders[i].empty()  # Clear the previous content
            with summary_placeholders[i].container():
                st.markdown(f"**Document: {doc_names[i]}**")
                with st.expander("View Summary"):
                    st.write(summary)
                with st.expander("View Full Content"):
                    st.write(doc_contents[i])

    # Once all summaries are done, move to step 4
    st.markdown(
        "**All documents summarized. Proceeding to final answer generation.**")

    # Step 4: Generate Answer Without Citation
    st.markdown("**4️⃣ Step 4: Generate Answer Without Citation**")
    # Placeholder for generating answer message
    generating_answer_placeholder = st.empty()
    generating_answer_placeholder.markdown("**Generating answer...**")

    final_answer = generate_final_answer(user_query, summaries)
    generating_answer_placeholder.empty()
    st.write("### 🗒️ **Final Answer (Before Citation):**")
    st.write(final_answer)
    st.write("")

    generating_answer_placeholder = st.empty()
    generating_answer_placeholder.markdown("**Adding citations...**")

    # Second LLM call for adding citations
    annotated_answer = add_citations_to_answer(final_answer, references)

    # Parse the citations
    citations = parse_citations(annotated_answer)

    # Replace citations with nice markdown links
    annotated_answer_with_links = replace_citations_with_links(
        annotated_answer)
    generating_answer_placeholder.empty()
    # Display the annotated answer
    st.markdown("## 📝 **Final Answer (Annotated with Citations):**")
    st.write("")

    st.markdown(annotated_answer_with_links)

    if citations:
        st.markdown("### References Found in Answer:")
        seen_citations = set()
        for c in citations:
            citation_key = (c['pr_number'], c['title'], c['url'])
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                st.markdown(
                    f"- PR #{c['pr_number']}: [{c['title']}]({c['url']})")
