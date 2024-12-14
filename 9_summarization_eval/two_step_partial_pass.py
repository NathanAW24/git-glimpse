import os
import json
from dotenv import load_dotenv
import evaluate
from pprint import pprint
import openai
from textwrap import wrap

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Generic LLM call function


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

# Function to read all .txt files in a folder


def read_txt_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    documents = []
    for file in files:
        with open(os.path.join(folder_path, file), "r") as f:
            documents.append(f.read())
    return documents

# Function to summarize using a specific GPT model


def summarize_with_gpt(model, text):
    system_prompt = "You are a helpful assistant specializing in summarization. Your summary need to be detailed as this is for coding. If there is code, make sure you include it. Your summary must be long and detailed"
    user_prompt = f"Summarize the following text:\n{text}"
    summary = call_llm(system_prompt, user_prompt,
                       model=model, temperature=0.7)
    return summary.strip()

# Two-step partial-pass summarization


def two_step_partial_pass(model, text, chunk_size=3000):
    """Divide the text into chunks, summarize each chunk, and combine the summaries."""
    # Split text into smaller chunks
    chunks = wrap(text, chunk_size)

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_summary = summarize_with_gpt(model, chunk)
        chunk_summaries.append(chunk_summary)

    # Combine chunk summaries into a final summary
    combined_text = " ".join(chunk_summaries)
    final_summary = summarize_with_gpt(model, combined_text)
    return final_summary

# Function to evaluate summaries using HaRiM+


def evaluate_summaries_harimplus(articles, summaries):
    scorer = evaluate.load("NCSOFT/harim_plus")
    scores = scorer.compute(predictions=summaries, references=articles)
    return scores

# Main script


def main():
    folder_path = "processed_docs"  # Folder containing the .txt files
    output_path = "summarization_results.json"  # JSON file to save results

    # Read the documents
    documents = read_txt_files(folder_path)
    if not documents:
        print("No documents found in the specified folder.")
        return

    # Prepare models
    gpt_models = ["gpt-4o-mini"]
    results = {}

    # Summarize and evaluate each model
    for model in gpt_models:
        print(f"Summarizing with {model} using two-step partial-pass...")
        model_results = []
        for i, doc in enumerate(documents):
            summary = two_step_partial_pass(model, doc)
            score = evaluate_summaries_harimplus([doc], [summary])
            model_results.append({
                "document_id": i + 1,
                "summary": summary,
                "score": score
            })
        results[model] = model_results
        pprint({f"{model}_results": model_results})

    # Save results to JSON file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
