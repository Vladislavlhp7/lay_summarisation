import os
from random import seed

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words


def load_article_dataset(dtype: str, filename: str, directory: str) -> Dataset:
    """
    Load an article dataset of a specified type from a given directory.

    Args:
    dtype (str): The type of dataset to load, such as 'train' or 'val'.
    filename (str): The name of the dataset file to load.
    directory (str): The directory path where the dataset file is located.

    Returns:
    Dataset: A Hugging Face Datasets object containing the loaded dataset.
    """

    # Construct the path to the dataset file
    path = os.path.join(directory, f"{dtype}/{filename}_{dtype}.jsonl")

    # Load the dataset into a Pandas DataFrame
    df = pd.read_json(path, lines=True)

    # Convert the DataFrame to a Hugging Face Datasets object
    dataset = Dataset.from_pandas(df)

    # Return the loaded dataset
    return dataset


def set_seed(seed_v: int = 42) -> None:
    """
    Set the random seed for the random number generators used by NumPy, Python, and PyTorch.

    Args:
    seed_v (int): The value to use as the random seed.

    Returns:
    None
    """

    # Set the random seed for NumPy
    np.random.seed(seed_v)

    # Set the random seed for Python
    seed(seed_v)

    # Set the random seed for PyTorch on the CPU and GPU
    torch.manual_seed(seed_v)
    torch.cuda.manual_seed(seed_v)

    # Set the hash seed to a fixed value for consistent hash values
    os.environ["PYTHONHASHSEED"] = str(seed_v)

    # Print a message to indicate the random seed has been set
    print(f"Random seed set as {seed_v}")


def read_jsonl_data(path):
    """
    Load data from the given path.
    :param path: path to the data
    :return: data
    """
    return pd.read_json(path, lines=True)


def create_article_dataset_dict(
    filename: str, directory: str, batch_size: int, tokenizer, max_input_length: int
) -> DatasetDict:
    """
    Create a dictionary of preprocessed datasets from article data in a given directory.

    Args:
        filename (str): The filename of the dataset to load.
        directory (str): The directory path where the dataset is located.
        batch_size (int): The batch size to use for processing the dataset.
        tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object to use for tokenization.
        max_input_length (int): The maximum length of the input and output sequences after tokenization.

    Returns:
        DatasetDict: A dictionary containing preprocessed datasets for training and validation.
    """

    # Define the dataset types to load
    dataset_types = ["train", "val"]

    # Initialize an empty dictionary to store the preprocessed datasets
    datasets = {}

    # Iterate through each dataset type and preprocess the data
    for dtype in dataset_types:
        # Load the dataset
        dataset = load_article_dataset(dtype, filename, directory)

        # Preprocess the data for model inputs
        dataset = dataset.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "lay_summary", "headings"],
            fn_kwargs={"tokenizer": tokenizer, "max_input_length": max_input_length},
        )

        # Set the format of the dataset to be used with PyTorch
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )

        # Add the preprocessed dataset to the datasets dictionary
        datasets[dtype] = dataset

    # Return the preprocessed datasets as a DatasetDict
    return DatasetDict(datasets)


def process_data_to_model_inputs(
    batch, tokenizer, max_input_length, presummarise=False
):
    """
    Tokenize and preprocess a batch of data for use as model inputs.

    Args:
    batch (dict): A dictionary containing the input and output data for the batch.
    tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object to use for tokenization.
    max_input_length (int): The maximum length of the input and output sequences after tokenization.
    presummarise (bool): Whether to presummarise the input data before tokenization.

    Returns:
    dict: A dictionary containing the preprocessed model inputs for the batch.
    """

    if presummarise:
        # Use LexRank to summarize the article
        article_summary = lexrank_summarize(batch["article"])
    else:
        article_summary = batch["article"]

    # Tokenize the inputs and outputs using the provided tokenizer
    inputs = tokenizer(
        article_summary,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["lay_summary"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )

    # Create a dictionary to store the preprocessed model inputs
    processed_batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

    # Assign the tokenized inputs and attention masks to the processed batch dictionary

    # Create a list of 0s to use as the global attention mask
    global_attention_mask = [
        [0] * len(processed_batch["input_ids"][0])
        for _ in range(len(processed_batch["input_ids"]))
    ]
    # Set the first element of the global attention mask to 1 to indicate the start of the sequence
    global_attention_mask[0][0] = 1
    processed_batch["global_attention_mask"] = global_attention_mask

    # Assign the tokenized outputs and label masks to the processed batch dictionary
    processed_batch["labels"] = outputs.input_ids
    # Replace the PAD tokens with -100 to ignore them during training
    processed_batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in processed_batch["labels"]
    ]

    # Return the preprocessed model inputs as a dictionary
    return processed_batch


def lexrank_summarize(article: str, sentence_count: int = 25) -> str:
    """
    Use LexRank to generate a summary of an article.

    Args:
    article (str): The text of the article to summarize.
    sentence_count (int, optional): The number of sentences to include in the summary.
                                    Defaults to 25 due to 75% of the lay summaries being shorter than 25 sentences.

    Returns:
    str: The summary generated by LexRank.
    """

    # Initialize a parser and tokenizer for the article
    parser = PlaintextParser.from_string(article, Tokenizer("english"))

    # Initialize a LexRank summarizer with English stop words
    summarizer = LexRankSummarizer()
    summarizer.stop_words = get_stop_words("english")

    # Use LexRank to score the sentences and select the top K sentences
    summary_sentences = summarizer(parser.document, sentence_count)

    # Join the summary sentences into a single string and return it
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary


if __name__ == "__main__":
    data = read_jsonl_data("data/task1_development/train/eLife_train.jsonl")
    summaries = lexrank_data(data.article.to_list()[:5])
    [print(s) for s in summaries[1]]
