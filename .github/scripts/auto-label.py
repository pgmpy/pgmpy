import os

from gliclass import GLiClassModel, ZeroShotClassificationWithChunkingPipeline
from transformers import AutoTokenizer

MODEL_ID = "knowledgator/gliclass-small-v1.0"

LABELS = [
    "Base Models",
    "Bug",
    "Causal Discovery",
    "Causal Identification",
    "Datasets",
    "Example Models",
    "IO",
    "Maintenance",
    "Metrics",
    "Parameter Estimation",
    "Performance",
    "Probabilistic Inference",
    "Simulations",
    "Tests",
    "Time Series",
]

PROMPT = (
    "Classify this pgmpy GitHub issue based on its title and body. "
    "Identify the primary technical topic or software module involved. "
    "Focus on the actual functionality being discussed rather than "
)

def classifier(title: str, body: str) -> str:
    model = GLiClassModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    chunking_pipeline = ZeroShotClassificationWithChunkingPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        device="cpu",
        text_chunk_size=8192,
        text_chunk_overlap=256,
        labels_chunk_size=8,
    )

    text = f"Title: {title}\n\nBody:\n{body}"

    return chunking_pipeline(text, LABELS, prompt=PROMPT)[0][0]["label"]


if __name__ == "__main__":
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ["ISSUE_BODY"]

    predicted_label = classifier(issue_title, issue_body)

    print(f"Predicted Label: {predicted_label}")

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"label={predicted_label}\n")
