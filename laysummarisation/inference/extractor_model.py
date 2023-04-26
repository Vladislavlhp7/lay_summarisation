from transformers import (TrainingArguments)
from model.extractor_model import generate_summary
from utils import remove_citations


def perform_inference(model, tokenizer, article: str, max_length: int = 512, top_k: int = 25):
    """
    Generate summaries from the BERT model

    Args:
        model (BertForSequenceClassification): The model.
        tokenizer (BertTokenizerFast): The tokenizer.
        article (str): The article to summarise.
        max_length (int): The maximum number of tokens to generate.
        top_k (int): The number of sentences to select.

    Returns:
        summary (str): The generated summary.
    """

    args = TrainingArguments(output_dir='tmp/')
    summary = generate_summary(model, tokenizer, article, max_length=max_length, top_k=top_k, args=args)
    summary = remove_citations(summary)
    return summary
