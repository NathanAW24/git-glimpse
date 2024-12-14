import os
from dotenv import load_dotenv
import openai
import json
from statistics import mean
from pydantic import BaseModel, Field
import glob

# Additional imports for metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import evaluate

# nltk.download('wordnet', quiet=True)  # Ensure wordnet is downloaded for METEOR

# Load a pre-trained language model for perplexity calculation
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_model.eval()

########################
# Metrics Implementations
########################


def compute_accuracy(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


def compute_bleu(pred: str, ref: str) -> float:
    ref_tokens = ref.strip().split()
    pred_tokens = pred.strip().split()
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens,
                         smoothing_function=smoothie)
    return float(bleu)


def compute_meteor(pred: str, ref: str) -> float:
    # Tokenize inputs before passing to meteor_score
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()
    meteor = meteor_score([ref_tokens], pred_tokens)
    return float(meteor)


def compute_bertscore(pred: str, ref: str) -> float:
    P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)
    return float(F1.mean().item())


def compute_perplexity(text: str, model_id: str = 'gpt2') -> float:
    """
    Compute the perplexity of a given text using a specified pre-trained language model.

    Args:
        text (str): The input text for which to compute perplexity.
        model_id (str): The name of the pre-trained model to use (default is 'gpt2').

    Returns:
        float: The computed perplexity score.
    """
    # Load the perplexity metric
    perplexity = evaluate.load("perplexity", module_type="measurement")

    # Compute perplexity
    results = perplexity.compute(
        model_id=model_id, add_start_token=True, data=[text])

    # Return the mean perplexity
    return results['mean_perplexity']

########################
# LLM Accuracy via Relevance Score
########################

# LangChain structured output setup for relevance scoring


class RelevanceScore(BaseModel):
    score: int = Field(
        description="The relevance score of the prediction from 0 to 10.")


model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(RelevanceScore)


def compute_accuracy_llm(pred: str, ref: str) -> float:
    """Use an LLM to compute the accuracy score."""
    prompt = f"""
You are tasked with evaluating the accuracy of a predicted answer compared to a reference answer. 
Provide a relevance score from 0 to 10 based on the following criteria:
Matching/Accurate = the information is correct (investigation + code to be examined + proposed solutions)

### Evaluation Context:
You are assessing whether the predicted answer matches the reference answer in terms of correctness and completeness. The evaluation focuses on:
- Identifying the **correct files** to look at (based on the described investigation and implementation).
- Proposing the **correct implementation steps** aligned with the reference structure.
- Following best practices such as documentation, clear instructions, and identifying possible pitfalls.

### Scoring Criteria:
Provide a score from 0 to 10 based on the following:
- **10**: The predicted answer is completely accurate, matching the reference answer in all aspects (files, implementation, and structure).
- **9**: The predicted answer is nearly accurate but misses minor details or contains slight inaccuracies.
- **8**: The predicted answer is mostly accurate but has noticeable gaps or deviations from the reference.
- **7**: The predicted answer is moderately accurate but lacks some significant details or introduces minor inaccuracies.
- **6**: The predicted answer is partially accurate but with limited alignment to the reference.
- **5**: The predicted answer is minimally accurate and omits critical details or introduces significant inaccuracies.
- **4**: The predicted answer is very inaccurate, with minimal alignment to the reference.
- **3**: The predicted answer is poorly accurate and has only a marginal connection to the reference.
- **2**: The predicted answer is barely accurate, with negligible connection to the reference.
- **1**: The predicted answer is almost entirely inaccurate, with no meaningful alignment to the reference.
- **0**: The predicted answer is completely inaccurate or off-topic, providing no useful information.

### Response Format:
Provide a single integer from 0 to 10 as the accuracy score. Do not include any explanation, text, or additional characters.

### Reference Answer:
{ref}

### Predicted Answer:
{pred}

Provide your accuracy score (0-10):
"""
    response = structured_llm.invoke(prompt)
    return response.score / 10.0  # Normalize to a 0-1 scale


########################
# LLM Call Function
########################


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_llm(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7):
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

########################
# Evaluation Results Class
########################


class EvaluationResults:
    def __init__(self):
        # list of dicts: each dict has {q_id, accuracy, bleu, meteor, bertscore, perplexity}
        self.results = []

    def add_result(self, q_id: int, pred: str, ref: str):
        """Compute metrics and store result."""
        accuracy = compute_accuracy_llm(pred, ref)
        bleu = compute_bleu(pred, ref)
        meteor = compute_meteor(pred, ref)
        bert = compute_bertscore(pred, ref)
        ppl = compute_perplexity(pred)

        self.results.append({
            "q_id": q_id,
            "accuracy": accuracy,
            "bleu": bleu,
            "meteor": meteor,
            "bertscore": bert,
            "perplexity": ppl
        })

    def to_json(self):
        """Print results per Q/A and the overall averages in JSON format."""
        if not self.results:
            return json.dumps({"results": [], "averages": {}}, indent=2)

        avg_accuracy = mean(r["accuracy"] for r in self.results)
        avg_bleu = mean(r["bleu"] for r in self.results)
        avg_meteor = mean(r["meteor"] for r in self.results)
        avg_bert = mean(r["bertscore"] for r in self.results)
        avg_ppl = mean(r["perplexity"] for r in self.results)

        data = {
            "results": self.results,
            "averages": {
                "accuracy": avg_accuracy,
                "bleu": avg_bleu,
                "meteor": avg_meteor,
                "bertscore": avg_bert,
                "perplexity": avg_ppl
            }
        }
        return json.dumps(data, indent=2)


########################
# Example Workflow
########################

# Assuming the folder structure:
# 10_answer_evaluation/
# ├── QnA
# │   ├── question_1.txt
# │   ├── answer_1.txt
# │   ├── question_2.txt
# │   ├── answer_2.txt
# ...
# ├── vanilla.py
#
# We will load each question and answer pair, call the LLM, and evaluate.

if __name__ == "__main__":
    qna_folder = "./QnA"
    question_files = sorted(
        glob.glob(os.path.join(qna_folder, "question_*.txt")))
    answer_files = sorted(glob.glob(os.path.join(qna_folder, "answer_*.txt")))

    # This prompt structure is what the LLM will see when generating answers:
    # (The instructions say: "For now just vanilla its ok. so no context provided just direct LLM.")
    # Example system prompt and user prompt from your instructions:
    system_prompt = "You are a helpful assistant that provides solutions. Follow instructions carefully."
    template = """
Now generate the answer in this format. 
(Do not put “purpose” again in your response, The purpose is for your reference.)

General Answer Structure
1. Investigation / Analysis
Purpose: Understand the problem or request thoroughly before attempting a fix or implementation.
* If it is FIXING ISSUE → identify what should be tested.
* Contextual Review: Summarize the scenario or requirement. For example, "We need to add virtualization to the NextUI Table component to handle large datasets efficiently."
* Inspection of Existing Code and Behavior: Mention which files, components, or logic currently handle the functionality. For example, "Check table.tsx and related hooks that manage the Table selection logic."
* Relevant Files (to look at)
    * DIFFERENTIATE THE NEW FILES YOU SHOULD CREATE OR THE FILES YOU SHOULD TAKE A LOOK AT. IN THIS SECTION YOU ONLY WANT THE FILES YOU SHOULD SEE AND NOT CREATE
    * checkbox.tsx: Handles rendering of checkboxes.
    * table.tsx: Manages table logic, including row selection.
    * use-checkbox.ts: Defines the checkbox component’s internal logic.
* Error and Performance Analysis: If it's a bug, detail the root cause. If it's a new feature or refactor, identify current limitations. For instance, "The issue may arise from focus handling in checkboxes or the current event listeners causing multiple click registrations."
2. Proposed Solution / Implementation Steps
Purpose: Outline the solution clearly, focusing on what changes need to be made and where.
* Files to create or add
* IN THIS SECTION YOU WANT THE FILES YOU SHOULD CREATE.
    * Create new file in this folder. this file purpose is to …
* High-Level Changes: A brief overview of the strategy, e.g., "Introduce virtualization using @tanstack/react-virtual and adjust selection logic to ensure single focus event triggers."
* Specific Technical Steps:
    * Dependencies: "Install or update @tanstack/react-virtual to enable virtualization."
    * Refactoring / Adding Features: "Refactor the checkbox rendering logic in checkbox.tsx to leverage a hiddenInput slot and ensure proper focus states."
    * Updating Rendering Logic: "In table.tsx, update how rows handle selection events, ensuring that clicking a checkbox triggers a single, controlled event."
    * Improving or Adding Properties: "Add a new prop, isVirtualized, to control virtualization behavior and ensure backward compatibility."
3. Post-Fix / Post-Implementation Checks
Purpose: Validate that the solution works and does not introduce regressions.
* update unit test if necessary
* Testing and Verification: "Run unit tests and user interaction tests to ensure that rows are selectable again and that no new focus issues arise."
* Performance and Reliability: "Check if virtualization improves rendering performance on large datasets and that no extra clicks are registered."
4. Documentation and Communication (if needed)
Purpose: Ensure that all changes are clearly communicated and easy to understand for future maintainers and users.
* Update Documentation: "Add usage examples and prop definitions for the new virtualization feature in docs/components/table.mdx."
* Add Release Notes: "In the project’s CHANGELOG, mention the new virtualization support and the fixed interaction bug."
"""

    evaluator = EvaluationResults()

    # Iterate over Q/A pairs
    for i, (q_file, a_file) in enumerate(zip(question_files, answer_files), start=1):
        with open(q_file, "r") as fq, open(a_file, "r") as fa:
            question = fq.read().strip()
            reference_answer = fa.read().strip()

        # Get predicted answer from LLM
        user_prompt = f"{question}\n\n{template}"
        predicted_answer = call_llm(
            system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7)

        # Store metrics
        evaluator.add_result(i, predicted_answer, reference_answer)

    # Print out results as JSON
    print(evaluator.to_json())
