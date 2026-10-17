import os


def classifier(title: str) -> str:
    # TODO: Replace with GLiClass classification model
    return "Bug"


if __name__ == "__main__":
    # Step 1. Get issue title
    issue_title = os.environ["ISSUE_TITLE"]

    print(f"Issue Title: {issue_title}")

    # Step 2. Classify issue
    predicted_label = classifier(issue_title)

    print(f"Predicted Label: {predicted_label}")

    # Step 3. Return label to GitHub Actions
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"label={predicted_label}\n")
