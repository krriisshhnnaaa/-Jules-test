import sys
from typing import List

# Suppress Hugging Face warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline
except ImportError:
    print("Error: The 'transformers' library is not installed. Please install it using 'pip install transformers torch'.")
    sys.exit(1)


def has_coherent_sentences(texts: List[str]) -> bool:
    """
    Predicts whether any text in the given list is a coherent sentence.

    Args:
        texts (List[str]): A list of texts to evaluate.

    Returns:
        bool: True if at least one text is coherent, False otherwise.
    """
    if not texts:
        return False

    # Use the CoLA (Corpus of Linguistic Acceptability) model
    classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    results = classifier(texts)

    # Check if any of the results are labeled as acceptable (LABEL_1)
    for result in results:
        if result['label'] == 'LABEL_1':
            return True

    return False

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "is a this sentence coherent not.",
        "The dog chased the cat.",
        "dog the chased cat the."
    ]

    print("Evaluating sample texts:")
    for text in sample_texts:
        print(f" - {text}")

    result = has_coherent_sentences(sample_texts)
    print(f"\nResult (Are there any coherent sentences?): {result}")
