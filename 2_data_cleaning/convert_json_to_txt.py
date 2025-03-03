import os
import json
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def process_pull_request(pr: Dict) -> str:
    """
    Convert a single pull request dictionary into a document string.
    """
    files_changed = pr.get('files_with_diffs', [])
    if not files_changed:
        files_changed_text = "No files changed."
    else:
        files_changed_text = ''.join([
            (
                f"- {file['filename']} (renamed){chr(92)}n"
                f"  Previous Filename: {file['previous_filename']}{chr(92)}n"
            ) if file['status'] == 'renamed' else (
                f"- {file['filename']} ({file['status']}, {file['changes']} changes){chr(92)}n"
                f"  Patch: {file.get('patch', 'No patch available')}{chr(92)}n"
            )
            for file in files_changed
        ])

    comments = pr.get('comments', {}).get('nodes', [])
    comments_text = ''.join([
        f"- {comment['author']['login']}: {comment['body']}\\n"
        for comment in comments
        if comment and comment.get('author') and comment['author'].get('login') and comment.get('body')
    ]) if comments else "No comments."

    document = f"""
Pull Request Number: {pr['number']}
Title: {pr['title']}
Base Branch: {pr['baseRefName']}
Head Branch: {pr['headRefName']}
Author: {pr['author']['login'] if pr['author'] else "Unknown"}
URL: {pr['url']}
State: {pr['state']}
Created At: {pr['createdAt']}
Merged At: {pr.get('mergedAt', 'Not Merged')}
Participants: {', '.join([p['login'] for p in pr.get('participants', {}).get('nodes', [])])}

Description:
{pr.get('bodyText', 'No description provided.')}

Commits:
{''.join([f"- {commit['commit']['message']}{chr(92)}n" for commit in pr.get('commits', {}).get('nodes', [])])}

Labels:
{', '.join([label['name'] for label in pr.get('labels', {}).get('nodes', [])])}

Comments:
{comments_text}

Files Changed:
{files_changed_text}
"""
    return document.strip()


def process_json_file(file_path: str) -> List[str]:
    """
    Read a JSON file and convert its pull request data to documents.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        pr_nodes = data.get('data', {}).get('repository', {}).get(
            'pullRequests', {}).get('nodes', [])
        return [process_pull_request(pr) for pr in pr_nodes]


def traverse_folder_and_convert_to_documents(folder_path: str, output_path: str):
    """
    Traverse the folder, process JSON files in order, and save documents to the output folder.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Sort files in the order of `pr_data_0.json` to `pr_data_207.json`
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for file in files:
        file_path = os.path.join(folder_path, file)
        logging.info(f"Processing file: {file_path}")
        documents = process_json_file(file_path)
        for idx, doc in enumerate(documents):
            output_file = os.path.join(
                output_path, f"{os.path.splitext(file)[0]}_doc_{idx}.txt")
            with open(output_file, 'w') as f:
                f.write(doc)


if __name__ == "__main__":
    input_folder = "json_pr_data_from_github"  # Folder containing JSON files
    output_folder = "processed_docs"  # Folder to save processed documents
    traverse_folder_and_convert_to_documents(input_folder, output_folder)
