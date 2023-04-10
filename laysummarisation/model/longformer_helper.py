from datasets import DatasetDict, Dataset
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
import numpy as np
from rouge import Rouge
from random import seed


def process_data_to_model_inputs(batch, tokenizer, max_input_length):
    """
    Tokenize and preprocess a batch of data for use as model inputs.
    
    Args:
    batch (dict): A dictionary containing the input and output data for the batch.
    tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object to use for tokenization.
    max_input_length (int): The maximum length of the input and output sequences after tokenization.
    
    Returns:
    dict: A dictionary containing the preprocessed model inputs for the batch.
    """

    # Tokenize the inputs and outputs using the provided tokenizer
    inputs = tokenizer(
        batch["article"],
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
    processed_batch = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

    # Assign the tokenized inputs and attention masks to the processed batch dictionary

    # Create a list of 0s to use as the global attention mask
    global_attention_mask = [[0] * len(processed_batch["input_ids"][0]) for _ in
                             range(len(processed_batch["input_ids"]))]
    # Set the first element of the global attention mask to 1 to indicate the start of the sequence
    global_attention_mask[0][0] = 1
    processed_batch["global_attention_mask"] = global_attention_mask

    # Assign the tokenized outputs and label masks to the processed batch dictionary
    processed_batch["labels"] = outputs.input_ids
    # Replace the PAD tokens with -100 to ignore them during training
    processed_batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                                 processed_batch["labels"]]

    # Return the preprocessed model inputs as a dictionary
    return processed_batch


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
    path = os.path.join(directory, f'{dtype}/{filename}_{dtype}.jsonl')

    # Load the dataset into a Pandas DataFrame
    df = pd.read_json(path, lines=True, nrows=100)

    # Convert the DataFrame to a Hugging Face Datasets object
    dataset = Dataset.from_pandas(df)

    # Return the loaded dataset
    return dataset


def create_article_dataset_dict(filename: str, directory: str, batch_size: int, tokenizer,
                                max_input_length: int) -> DatasetDict:
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
    dataset_types = ['train', 'val']

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
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'global_attention_mask', 'labels'])

        # Add the preprocessed dataset to the datasets dictionary
        datasets[dtype] = dataset

    # Return the preprocessed datasets as a DatasetDict
    return DatasetDict(datasets)


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


def compute_metrics(eval_pred) -> dict:
    """
    Compute the ROUGE scores for a given set of predictions and labels.
    
    Args:
    eval_pred (tuple): A tuple containing two lists of predictions and labels.
    
    Returns:
    dict: A dictionary containing the ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """

    # Unpack the tuple into separate lists of predictions and labels
    predictions, labels = eval_pred

    # Compute the ROUGE scores for the predictions and labels using the Rouge package
    rouge = Rouge()
    scores = rouge.get_scores(predictions, labels, avg=True)

    # Return the ROUGE scores as a dictionary with keys for each metric
    return {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"]
    }