import streamlit as st
import re
import os
from dotenv import load_dotenv
from retriever import EnsembleRetriever
import openai

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
# The goal of this Streamlit app is to demonstrate an enhanced version of RAG:
# 1. User enters a question (e.g., how to enhance a component).
# 2. The LLM refines the query before retrieval.
# 3. Relevant documents (PRs) are retrieved from BM25 and vector search.
# 4. Each document is summarized to reduce length.
# 5. The final answer is generated using all summarized contexts.
#
# The system prompt should ensure that the assistant:
# - Acts as a senior developer or mentor guiding a newcomer developer.
# - Provides detailed instructions on how to onboard and understand the codebase.
# - Suggests where to find files, what naming conventions to follow, what utilities can be leveraged.
# - Highlights learnings from previous PRs, common pitfalls, and best practices.
# - In short, it should produce a final answer that helps a new developer quickly get productive.

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
   - Summarize each relevant document to reduce length.
   - Use the summarized content to generate a final answer that helps you onboard quickly.
3. You will see the chain-of-thought process, the final answer, and any relevant PR links.
   
**Developer-Oriented Guidance:**
- The final answer will not just answer the question directly, but also:
  - Suggest which files or directories to look into.
  - Mention coding conventions or patterns used in the codebase.
  - Suggest libraries or utilities already integrated, so you don't reinvent the wheel.
  - Highlight insights from PR comments or commits that show how previous developers solved similar issues.
  - Outline common pitfalls and best practices.
""")

# Updated prompt templates with system and user messages
PROMPT_TEMPLATES = {
    "query_refinement": {
        "system": """You are a senior developer at a large company. Your job is to refine and expand any given user query related to a software codebase or feature. The user is a newcomer developer who needs more specific, detailed, and actionable queries to find the right documents. Your refined query should:
- Clarify ambiguities.
- Add relevant keywords or technologies that might be involved.
- Ensure the query is well-targeted so that the retrieval system can find the most pertinent PRs or code references.
""",
        "user": """Original user query: {query}\nPlease refine and expand this query to improve document retrieval."""
    },
    "summarization": {
        "system": """You are a senior developer summarizing a document (e.g., a PR). The summary should:
- Focus on the context relevant to the given query.
- Highlight which files or modules are touched.
- Mention coding conventions, library usage, or patterns observed.
- Note any pitfalls, best practices, or insights from PR discussions.
- Identify if there are utilities or frameworks already used, so a new developer doesn't have to re-invent solutions.
If the context does not provide you with a relevant topics - skip explaining about that topics.
Your answer must be concise.
The goal is to give a new developer actionable insights to quickly understand and leverage the work done in this PR.
""",
        "user": """Query: {query}\nDocument Content:\n{content}\nSummarize the above document with these objectives in mind."""
    },
    "final_answer": {
        "system": """You are a senior developer synthesizing the final answer for a newcomer developer. You have several summarized documents (PRs) that relate to the user's question. Your final answer should:
- Directly answer the user's question.
- Suggest files or directories to look at.
- Highlight relevant coding patterns, naming conventions, or established best practices found in the PRs.
- Suggest libraries or existing utilities that can be leveraged.
- Point out common pitfalls and how previous developers addressed them.
- Include why certain decisions were made and how to align with the project's standards.
If the context does not provide you with a relevant topics - skip explaining about that topics.
Your answer must be detailed but using easy to understand and concise language.
This is about empowering the new developer to ramp up quickly and make informed decisions.
""",
        "user": """User question: {query}\nContext from Summarized Documents:\n{context}\nProvide the best possible answer, incorporating all the mentioned developer onboarding guidance."""
    }
}


def call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0.7):
    """Generic LLM call using OpenAI API with a system and a user prompt."""
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
    """Refines and expands the user's query."""
    system_prompt = PROMPT_TEMPLATES["query_refinement"]["system"]
    user_prompt = PROMPT_TEMPLATES["query_refinement"]["user"].format(
        query=query)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


def summarize_document(query, content):
    """Summarizes the given document content from a developer onboarding perspective."""
    system_prompt = PROMPT_TEMPLATES["summarization"]["system"]
    user_prompt = PROMPT_TEMPLATES["summarization"]["user"].format(
        query=query, content=content)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


def generate_final_answer(query, summaries):
    """Generates the final onboarding-focused answer based on query and summaries."""
    context = "\n\n".join(
        [f"Document {i+1} summary:\n{summ}" for i, summ in enumerate(summaries)])
    system_prompt = PROMPT_TEMPLATES["final_answer"]["system"]
    user_prompt = PROMPT_TEMPLATES["final_answer"]["user"].format(
        query=query, context=context)
    return call_llm(system_prompt, user_prompt, model="gpt-4o", temperature=0)


def extract_pr_info(text):
    """Extracts PR number, title, and URL from the given text."""
    pr_number_pattern = r"Pull Request Number:\s*(\d+)"
    pr_title_pattern = r"Title:\s*(.*)"
    pr_url_pattern = r"URL:\s*(https?://[^\s]+)"

    pr_num = re.search(pr_number_pattern, text)
    pr_title = re.search(pr_title_pattern, text)
    pr_url = re.search(pr_url_pattern, text)

    if pr_num and pr_url:
        return pr_num.group(1), (pr_title.group(1) if pr_title else "No Title Found"), pr_url.group(1)
    return None


# Streamlit Workflow
user_query = st.text_input(
    "Ask a question:", "How can I enhance the validation capabilities of a Select component?")
run_button = st.button("Run")

if run_button:
    st.subheader("Chain of Thought")

    # Step 1: Query Refinement
    st.markdown("**Step 1: Query Refinement**")
    refined_query = prompt_clean_and_expand(user_query)
    st.write("**Refined Query:**", refined_query)

    # Step 2: Retrieve Relevant Documents
    st.markdown("**Step 2: Retrieve Documents**")
    retrieved_docs = retriever.get_documents(refined_query)

    # Display PR info for each retrieved document
    if retrieved_docs:
        for fname, content in retrieved_docs.items():
            pr_info = extract_pr_info(content)
            if pr_info:
                pr_num, pr_title, pr_url = pr_info
                st.markdown(f"- **PR #{pr_num}**: [{pr_title}]({pr_url})")
            else:
                st.markdown(f"- **Document:** {fname} (No PR info found)")
    else:
        st.write("No documents found.")

    # Step 3: Summarize Documents and Extract PR Info
    st.markdown("**Step 3: Summarize Documents and Extract PR Info**")
    summaries = []
    for fname, content in retrieved_docs.items():
        summary = summarize_document(refined_query, content)
        summaries.append(summary)
        st.markdown(f"**Document: {fname}**")
        with st.expander("View Summary"):
            st.write(summary)

        # Extract PR info again for display (already done above, but here we show inline if needed)
        prs = extract_pr_info(content)
        if prs:
            pr_num, pr_title, pr_url = prs
            st.markdown("**Relevant PR:**")
            st.markdown(f"- **PR #{pr_num}:** [{pr_title}]({pr_url})")
        else:
            st.write("No PRs found in this document.")

        # Collapsible full content
        with st.expander("View Full Content"):
            st.write(content)

    # Step 4: Generate Final Answer
    st.markdown("**Step 4: Generate Final Answer**")
    final_answer = generate_final_answer(user_query, summaries)
    st.write("**Final Answer:**", final_answer)
