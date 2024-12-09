import os
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Argument to control whether to use the local database or re-index
USE_LOCAL_DB = False  # Set to True to use existing local database, False to re-index

# Directory paths
# Folder containing processed PR data files in .txt format
PR_FOLDER = "processed_docs"
# Directory to save/load the vector database
VECTOR_DB_DIR = "1_gte_small_demo_hanjin"

# Model name for HuggingFace embeddings
MODEL_NAME = "thenlper/gte-small"

# Function to load PR data


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

# Example query function


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
