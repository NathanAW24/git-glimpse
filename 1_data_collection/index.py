import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

# GitHub GraphQL endpoint
GRAPHQL_URL = "https://api.github.com/graphql"
GRAPHQL_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}

# REST API headers
REST_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# GraphQL query with pagination
QUERY = """
query ($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $first, after: $after, states: [OPEN, MERGED, CLOSED], orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        bodyText
        baseRefName
        headRefName
        author {
          login
        }
        url
        state
        commits(first: 50) {
          nodes {
            commit {
              message
            }
          }
        }
        labels(first: 50) {
          nodes {
            name
          }
        }
        comments(first: 50) {
          nodes {
            body
            author {
              login
            }
          }
        }
        reviews(first: 50) {
          nodes {
            bodyText
            comments(first: 50) {
              nodes {
                body
                diffHunk
                author {
                  login
                }
              }
            }
          }
        }
        createdAt
        mergedAt
        participants(first: 50) {
          nodes {
            login
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

# Load saved cursors from a JSON file


def load_saved_cursors(file_path="cursors.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

# Save cursors to a JSON file


def save_cursors(cursors, file_path="cursors.json"):
    with open(file_path, "w") as file:
        json.dump(cursors, file, indent=4)


def fetch_pr_data(start_file_index=0):
    # Load saved cursors
    cursors = load_saved_cursors()
    rate_limit_retry_time = 60  # Retry after 1 minute on rate limit
    first = 50  # Number of PRs to fetch per query
    file_index = start_file_index
    after_cursor = cursors.get(str(file_index), None)

    # Create folder for storing JSON files
    output_folder = "pr_data"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Starting from file index: {file_index}")
    print(f"Loaded after_cursor: {after_cursor}")

    while True:
        try:
            print(f"\nQuerying PR data for file index: {file_index}")
            if after_cursor:
                print(f"Continuing from cursor: {after_cursor}")
            else:
                print("Starting from the beginning (no cursor).")

            # Define query variables
            variables = {
                "owner": REPO_OWNER,
                "name": REPO_NAME,
                "first": first,
                "after": after_cursor
            }

            # Send request to GitHub GraphQL API
            response = requests.post(
                GRAPHQL_URL,
                headers=GRAPHQL_HEADERS,
                json={"query": QUERY, "variables": variables}
            )

            if response.status_code == 200:
                data = response.json()
                # CMMETN THIS
                # with open(os.path.join(
                #         output_folder, f"pr_data_{file_index}.json") + "_temp", "w") as json_file:
                #     json.dump(data, json_file, indent=4)

                pr_data = data["data"]["repository"]["pullRequests"]
                print(
                    f"Successfully fetched PR data for file index: {file_index}")

                # Process each PR and fetch additional file diffs
                for pr in pr_data["nodes"]:
                    pr_number = pr["number"]
                    print(f"Processing PR #{pr_number}")

                    files_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files"
                    print(f"Fetching files with diffs from: {files_url}")

                    files_response = requests.get(
                        files_url, headers=REST_HEADERS)

                    if files_response.status_code == 200:
                        files_data = files_response.json()
                        print(
                            f"Successfully fetched files for PR #{pr_number}")
                        pr["files_with_diffs"] = files_data
                    else:
                        print(
                            f"Error fetching files for PR #{pr_number}: {files_response.status_code}")
                        pr["files_with_diffs"] = None

                # Save results to a JSON file inside the folder
                file_name = os.path.join(
                    output_folder, f"pr_data_{file_index}.json")
                with open(file_name, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Saved PR data to {file_name}")

                # Update cursor for the current file index
                if pr_data["pageInfo"]["hasNextPage"]:
                    after_cursor = pr_data["pageInfo"]["endCursor"]
                    print(
                        f"End cursor for file index {file_index}: {after_cursor}")
                    cursors[str(file_index + 1)] = after_cursor
                    save_cursors(cursors)
                    print(
                        f"Cursor saved for next file index: {file_index + 1}")
                    file_index += 1
                else:
                    print("No more pages available. All PRs fetched successfully!")
                    break
            elif response.status_code == 403 and "rate limit" in response.text.lower():
                print("Rate limit reached. Retrying after delay...")
                # time.sleep(rate_limit_retry_time)
            else:
                print(f"Error fetching PR data: {response.status_code}")
                print(response)
                print(response.json())
                break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break


# Main execution
if __name__ == "__main__":
    fetch_pr_data(5)
