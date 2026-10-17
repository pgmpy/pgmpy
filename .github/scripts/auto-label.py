import os

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MODEL_ID = "knowledgator/gliclass-small-v1.0"

LABELS = [
    "Base Models",
    "Bug",
    "Causal Discovery",
    "Causal Identification",
    "Datasets",
    "Documentation",
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


def classifier(title: str) -> str:
    model = GLiClassModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        device="cpu",
    )

    return pipeline(title, LABELS)[0][0]["label"]


if __name__ == "__main__":
    predicted_label = classifier(os.environ["ISSUE_TITLE"])

    print(f"Predicted Label: {predicted_label}")

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"label={predicted_label}\n")
