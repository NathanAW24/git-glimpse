import streamlit as st
import re
import os
from dotenv import load_dotenv
from retriever import EnsembleRetriever
import openai

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
   - Use the summarized content to generate a final answer.
3. You will see the chain-of-thought process and final answer. Also, if relevant, any PR links found in the documents will be displayed.
""")

user_query = st.text_input(
    "Ask a question:", "How can I enhance the validation capabilities of a Select component?")
run_button = st.button("Run")


def call_llm(messages, model="gpt-4o", temperature=0.7):
    # Generic LLM call using OpenAI API
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return str(response.choices[0].message.content)


def prompt_clean_and_expand(query):
    # Step 1: Clean and expand query
    system_msg = {"role": "system",
                  "content": "You are a helpful assistant that refines user queries to be more precise and detailed before retrieval."}
    user_msg = {"role": "user",
                "content": f"Original user query: {query}\nRefine and expand this query for better document retrieval."}
    refined_query = call_llm([system_msg, user_msg],
                             model="gpt-4o", temperature=0)
    return refined_query.strip()


def summarize_document(doc_content):
    # Summarize a single document
    system_msg = {"role": "system",
                  "content": "You are a helpful assistant that summarizes documents into concise paragraphs."}
    user_msg = {"role": "user",
                "content": f"Summarize the following document:\n{doc_content}"}
    summary = call_llm([system_msg, user_msg], model="gpt-4o", temperature=0)
    return summary.strip()


def generate_final_answer(user_query, summaries):
    # Given the user query and summarized docs, produce the final answer
    system_msg = {"role": "system",
                  "content": "You are a helpful assistant that uses the provided summarized context to answer the question accurately."}
    context = "\n\n".join(
        [f"Document {i+1} summary:\n{summ}" for i, summ in enumerate(summaries)])
    user_msg = {"role": "user",
                "content": f"User question: {user_query}\n\nUse the above context to provide the best possible answer."}
    answer = call_llm([system_msg, user_msg], model="gpt-4o", temperature=0)
    return answer.strip()


def extract_pr_info(text):
    # Use regex to extract PR number and URL
    # Example PR format given:
    # Pull Request Number: 4140
    # URL: https://github.com/nextui-org/nextui/pull/4140
    pr_pattern = r"Pull Request Number:\s*(\d+).*?URL:\s*(https?://[^\s]+)"
    matches = re.findall(pr_pattern, text, flags=re.DOTALL)
    return matches


if run_button:
    st.subheader("Chain of Thought")

    # Step 1: Prompt Cleaning and Query Expansion
    st.markdown("**Step 1: Query Refinement**")
    refined_query = prompt_clean_and_expand(user_query)
    st.write("**Refined Query:**", refined_query)

    # Step 2: Retrieve Relevant Documents
    st.markdown("**Step 2: Retrieve Documents**")
    retrieved_docs = retriever.get_documents(refined_query)
    st.write("**Retrieved Documents:**", list(retrieved_docs.keys()))

    # Step 3: Summarize Each Document and Extract PR Info
    st.markdown("**Step 3: Summarize Documents and Extract PR Info**")
    summaries = []
    for fname, content in retrieved_docs.items():
        summary = summarize_document(content)
        summaries.append(summary)
        st.markdown(f"**Document: {fname}**")
        st.write("**Summary:**", summary)

        # Extract PR info
        prs = extract_pr_info(content)
        if prs:
            st.markdown("**Relevant PRs:**")
            for pr_num, pr_url in prs:
                st.markdown(f"- PR #{pr_num}: [View on GitHub]({pr_url})")
        else:
            st.write("No PRs found in this document.")

        # Collapsible full content
        with st.expander("View Full Content"):
            st.write(content)

    # Step 4: Generate Final Answer
    st.markdown("**Step 4: Generate Final Answer**")
    final_answer = generate_final_answer(user_query, summaries)
    st.write("**Final Answer:**", final_answer)
