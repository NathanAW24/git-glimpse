import random
import os
import openai
import json

# Configuration Variables
# Replace with your OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Update with the directory containing PR files
BASE_PATH = "processed_docs"
RESULT_JSON_FILE = "questions_for_PR.json"     # Output file for storing results
NUM_SAMPLES = 10                     # Number of random samples to generate

# Set OpenAI API key
client = openai.OpenAI()


def generate_random_samples():
    """Generate NUM_SAMPLES random samples in the format pr_data_{x}_doc_{y}.txt."""
    samples = []
    for _ in range(NUM_SAMPLES):
        x = random.randint(0, 32)  # Range for 'pr_data_x'
        y = random.randint(1, 49)  # Range for 'doc_y'
        samples.append(f"pr_data_{x}_doc_{y}.txt")
    return samples


def read_pr_data(file_path):
    """Reads content from the PR file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return None


def generate_question(pr_content):
    """Generates a developer-style question using ChatGPT API."""
    prompt = f"""
    Imagine you are a developer trying to understand or learn something specific for your work.
    Your boss suggests looking at the following Pull Request to understand it better:

    {pr_content}

    Your question should not be asking about what are the code changes or what is the PR about.
    Your question is more like -> you are given a ticket from PM (feature or bug or refactor) -> assume one of this -> what would you ask if you want to start coding.
    Normally a developer will be start querying the old PRs to see how others have done it or not.
    
    Write the exact question you would ask that, if answered, the answer would directly be the content of this PR.
    The question should reflect a developer's perspective and be highly relevant to the PR's content.
    BE CONCISE. MAX 1 SENTENCE with few words.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant simulating a developer's thought process."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        print(response)
        return str(response.choices[0].message.content)
    except Exception as e:
        return f"Error: {e}"


def process_samples(samples, base_path):
    """Processes the sample PR files and generates developer questions."""
    results = []
    i = 0
    for sample in samples:
        print("Geneerating", i)
        file_path = os.path.join(base_path, sample)
        pr_content = read_pr_data(file_path)
        if pr_content:
            question = generate_question(pr_content)
            results.append({"file": sample, "question": question})
        else:
            results.append(
                {"file": sample, "question": "File not found or empty."})
        i += 1
    return results


# Main Execution
if __name__ == "__main__":
    # Generate random samples
    samples = generate_random_samples()

    # Process the samples to generate questions
    results = process_samples(samples, BASE_PATH)

    # Transform results to only include questions as a list of strings
    questions_list = [[result["file"], result["question"]]
                      for result in results]

    # Save results to a JSON file
    with open(RESULT_JSON_FILE, "w") as json_file:
        json.dump(questions_list, json_file, indent=4)

    # Output to console for verification
    for result in results:
        print(f"File: {result['file']}\nQuestion: {result['question']}\n")
