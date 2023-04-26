import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)
from laysummarisation.utils import (sentence_tokenize, preprocess)


def generate_summary(model, tokenizer, article: str, max_length: int = 512, top_k: int = 25, args: TrainingArguments = None):
    """
    Generate summary from the BERT model

    Args:
        model (BertForSequenceClassification): The model.
        tokenizer (BertTokenizerFast): The tokenizer.
        article (str): The article to summarise.
        max_length (int): The maximum number of tokens to generate.
        top_k (int): The number of sentences to select.
        args: The input arguments to the Trainer.

    Returns:
        summary (str): The generated summary.
    """
    # Prepare article for sentence tokenization
    article_cleaned = preprocess(article)
    # Segment article into sentences
    article_segmented = sentence_tokenize(article_cleaned)
    # Convert sentences to dataset
    article_df = pd.DataFrame({"sentence": article_segmented})
    article_dataset = Dataset.from_pandas(article_df)
    # Tokenize article sentences
    article_dataset = article_dataset.map(
        lambda x: tokenizer(x["sentence"], padding=True, truncation=True, return_tensors="pt",
                            max_length=max_length),
        batched=True,
        batch_size=1000,
    )
    # Convert to tensors
    article_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    # Generate summarising sentence probabilities
    trainer = Trainer(model=model, args=args)
    model.eval()
    with torch.no_grad():
        outputs = trainer.predict(article_dataset)
        predictions = torch.tensor(softmax(outputs.predictions, axis=1))
        # Sort probabilities based on the second element per list in descending order
        sorted_indices = torch.argsort(predictions[:, 1], descending=True)
        # Filter probabilities >= 0.5
        filtered_indices = sorted_indices[predictions[sorted_indices, 1] >= 0.5]
        # Take Top-K sentences in natural order where predictions are > 0.5
        top_k_indices = filtered_indices[:top_k]
        # Sort indices in ascending order
        top_k_indices = top_k_indices.sort().values
        top_k_sentences = [article_segmented[i] for i in top_k_indices]
        summary = " ".join(top_k_sentences)
    # trim summary to max_length words
    # summary = " ".join(summary.split()[:max_length])
    return summary
