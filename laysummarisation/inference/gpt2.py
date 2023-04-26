from transformers import (TrainingArguments)
from laysummarisation.model.gpt2 import generate_summary 

def gpt_summary(model, tokenizer, article: str, max_length: int = 1024):
    """
    Generate summaries from the GPT2 model

    Args:
        model: The model.
        tokenizer: The tokenizer.
        article (str): The article to summarise.
        max_length (int): The maximum number of tokens to generate.
        top_k (int): The number of sentences to select.

    Returns:
        summary (str): The generated summary.
    """

    args = TrainingArguments(output_dir='tmp/')
    summary = generate_summary(model, tokenizer, article, max_length=max_length, args=args)
    return summary
