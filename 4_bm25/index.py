import os
import math
from collections import Counter


class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        """
        Initialize BM25 Retriever.

        :param documents: List of documents (each document is a string).
        :param k1: BM25 parameter, usually in the range [1.2, 2.0].
        :param b: BM25 parameter, usually around 0.75.
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(documents)
        self.term_frequencies = [Counter(doc.split()) for doc in documents]
        self.document_frequencies = self._calculate_document_frequencies()

    def _calculate_document_frequencies(self):
        """
        Calculate document frequencies for all terms.

        :return: Dictionary of term to document frequency.
        """
        df = Counter()
        for term_freq in self.term_frequencies:
            for term in term_freq.keys():
                df[term] += 1
        return df

    def _idf(self, term):
        """
        Compute Inverse Document Frequency (IDF) for a term.

        :param term: The term for which IDF is computed.
        :return: IDF score.
        """
        n = len(self.documents)
        df = self.document_frequencies.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query, doc_index):
        """
        Calculate BM25 score for a document and a query.

        :param query: List of query terms.
        :param doc_index: Index of the document in the corpus.
        :return: BM25 score.
        """
        score = 0
        doc_length = self.doc_lengths[doc_index]
        term_freq = self.term_frequencies[doc_index]
        for term in query:
            tf = term_freq.get(term, 0)
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * \
                (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (numerator / denominator)
        return score

    def retrieve(self, query, top_n=5):
        """
        Retrieve the top N documents for a query based on BM25 scores.

        :param query: List of query terms.
        :param top_n: Number of top documents to return.
        :return: List of tuples (document_index, score).
        """
        query = query.split()  # Tokenize query
        scores = [(i, self.score(query, i))
                  for i in range(len(self.documents))]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


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
                    documents.append(f.read())
    return documents


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
    query = "floating components"
    top_n = 5  # Retrieve top 5 documents

    # Retrieve documents based on BM25 scores
    results = retriever.retrieve(query, top_n=top_n)

    # Display results
    for index, score in results:
        print(f"Document {index}: Score = {score}")
        # Display first 200 chars of the document
        print(f"Content: {documents[index][:200]}...\n")
