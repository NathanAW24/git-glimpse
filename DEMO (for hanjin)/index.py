import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def prompt_with_default(prompt, default):
    """
    Prompt the user for input, showing a default value.

    Args:
        prompt (str): The question to display to the user.
        default (str): The default value to use if the user provides no input.

    Returns:
        str: User's input or the default value if no input is provided.
    """
    user_input = input(f"{prompt} (Default: {default}): ").strip()
    return user_input if user_input else default


def confirm_with_default(prompt, default):
    """
    Prompt the user with a yes/no question, showing a default value.

    Args:
        prompt (str): The question to display to the user.
        default (str): Default response ('y' or 'n').

    Returns:
        bool: True if the user confirms ('y'), False otherwise.
    """
    default = default.lower()
    user_input = input(
        f"{prompt} (y/n, Default: {default}) - Press Enter to use default: ").strip().lower()
    if not user_input:
        user_input = default
    return user_input == 'y'


# Prompt the user to decide whether to use the local database
USE_LOCAL_DB = confirm_with_default(
    "Do you want to use the existing local database? If this is your first run, answer 'n'",
    "n"
)

# Prompt for directory paths and model name with default values
PR_FOLDER = prompt_with_default(
    "Enter the path to the folder containing processed PR data files",
    "processed_docs"
)
VECTOR_DB_DIR = prompt_with_default(
    "Enter the directory path to save/load the vector database",
    "1_gte_small_demo_hanjin"
)
MODEL_NAME = prompt_with_default(
    "Enter the model name for HuggingFace embeddings",
    "thenlper/gte-small"
)


def load_pr_data(pr_folder):
    """
    Load PR data from the specified folder.

    Args:
        pr_folder (str): Path to the folder containing PR data in .txt format.

    Returns:
        list[Document]: A list of Document objects, each containing the content of a PR and metadata.
    """
    pr_documents = []
    for file in os.listdir(pr_folder):
        if file.endswith(".txt"):  # Process only .txt files
            with open(os.path.join(pr_folder, file), 'r') as f:
                content = f.read()
                # Create a Document object for each file with metadata
                pr_documents.append(
                    Document(
                        page_content=content,
                        metadata={"file_name": file}
                    )
                )
    return pr_documents


# Main logic
if not USE_LOCAL_DB:
    # Re-indexing mode: Load PR data and create a new vector database
    print("Loading and processing PR data...")
    pr_documents = load_pr_data(PR_FOLDER)  # Load PR documents from the folder

    print("Generating embeddings and storing them in a new vector database... (Wait for few minutes)")
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        # Adjust device as needed ("cpu", "cuda", or "mps")
        model_kwargs={"device": "mps"},
        # Normalize embeddings for better similarity matching
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create a new Chroma vector store from the PR documents
    vectorstore = Chroma.from_documents(
        documents=pr_documents,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR  # Directory to store the vector database
    )

    # Save the vector database for future use
    vectorstore.persist()
    print(f"Vector database created and saved at {VECTOR_DB_DIR}")
else:
    # Use existing local vector database
    print("Using existing local vector database...")
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        # Ensure the device matches previous settings
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load the existing vector database
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model
    )
    print(f"Loaded vector database from {VECTOR_DB_DIR}")

# Retriever setup
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})


def query_prs(query):
    """
    Retrieve and display relevant PRs for a given query.

    Args:
        query (str): User's input query.

    Returns:
        list[dict]: List of dictionaries containing retrieved PR content and metadata.
    """
    print(f"Querying PRs for: {query}")
    retrieved_docs = retriever.invoke(query)  # Retrieve relevant documents
    results = [{"content": doc.page_content, "metadata": doc.metadata}
               for doc in retrieved_docs]
    return results


# Example usage
if __name__ == "__main__":
    example_query = "How do I create a new form on Angular v2?"
    relevant_prs = query_prs(example_query)
    print("Relevant PRs:")
    for idx, pr in enumerate(relevant_prs, start=1):
        print(f"PR {idx}:")
        # Show first 200 characters
        print(f"Content: {pr['content'][:200]}...")
        print(f"Metadata: {pr['metadata']}\n")
