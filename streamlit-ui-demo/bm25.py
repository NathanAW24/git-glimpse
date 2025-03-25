import os
from rank_bm25 import BM25Okapi


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


# Example Usage
if __name__ == "__main__":
    # Replace with the path to your folder containing .txt files
    folder_path = "processed_docs"

    # Load documents from the folder
    documents = _read_documents_from_folder(folder_path)
    print(f"Loaded {len(documents)} documents.")

    # Initialize BM25 retriever
    retriever = BM25Retriever(documents)

    # Query
    query = "performance"
    top_n = 5  # Retrieve top 5 documents

    # Retrieve documents based on BM25 scores
    results = retriever.retrieve(query, top_n=top_n)

    # Display results
    for index, score in results:
        print(f"Document {index}: Score = {score}")
        # Display first 200 chars of the document
        print(f"Content: {documents[index][:200]}...\n")
