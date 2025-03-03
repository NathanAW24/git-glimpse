import os
import json
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _read_documents_from_folder(folder_path):
    """
    Private utility function to read all .txt files from a folder.

    :param folder_path: Path to the folder containing text files.
    :return: List of document contents.
    """
    documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append([file, f.read()])
    return documents


class BM25Retriever:
    def __init__(self, documents):
        """
        Initialize BM25 Retriever using rank-bm25 library.

        :param documents: List of documents (each document is a string).
        """
        self.tokenized_documents = [doc.split()
                                    for [file_name, doc] in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def retrieve(self, query, top_n=5):
        """
        Retrieve the top N documents for a query based on BM25 scores.

        :param query: Query string.
        :param top_n: Number of top documents to return.
        :return: List of tuples (document_index, score).
        """
        tokenized_query = query.split()  # Tokenize query
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(index, scores[index]) for index in top_indices]


class EnsembleRetriever:
    def __init__(self, folder_path, vector_db_dir, model_name, device="mps"):
        # Load environment variables
        load_dotenv()

        # Initialize documents
        self.folder_path = folder_path
        self.documents = _read_documents_from_folder(folder_path)
        self.doc_dict = {doc[0]: doc[1] for doc in self.documents}

        # Initialize BM25 Retriever
        self.bm25_retriever = BM25Retriever(self.documents)

        # Initialize Vector Retrieval
        self.vectorstore = self._init_vectorstore(
            vector_db_dir, model_name, device)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def _init_vectorstore(self, vector_db_dir, model_name, device):
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        return Chroma(
            persist_directory=vector_db_dir,
            embedding_function=embedding_model
        )

    def get_documents(self, query):
        """Retrieve documents using ensemble retrieval mechanism."""
        # Retrieve from BM25
        bm25_results = self.bm25_retriever.retrieve(query, top_n=5)
        bm25_docs = [(self.documents[doc_index][0], self.documents[doc_index][1])
                     for doc_index, _ in bm25_results]

        # Retrieve from Vector Retrieval
        vector_results = self.vector_retriever.invoke(query)
        vector_docs = [(doc.metadata["file_name"], doc.page_content)
                       for doc in vector_results]

        # Combine results, removing duplicates by file name
        combined_docs = {}
        for fname, content in bm25_docs + vector_docs:
            if fname not in combined_docs:
                combined_docs[fname] = content

        return combined_docs


# Usage example
if __name__ == "__main__":
    folder_path = "processed_docs"
    vector_db_dir = "vector_db"
    model_name = "thenlper/gte-small"

    retriever = EnsembleRetriever(folder_path, vector_db_dir, model_name)

    query = "How can I enhance the validation capabilities of a Select component?"
    results = retriever.get_documents(query)

    # Print results
    for file_name, content in results.items():
        print(f"File: {file_name}\nContent: {content}\n")
