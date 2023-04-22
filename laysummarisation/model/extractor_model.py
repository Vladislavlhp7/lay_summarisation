from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset, DatasetDict, Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoModel, BertForSequenceClassification, BertTokenizerFast
import torch
import os
import numpy as np
import pandas as pd
from laysummarisation.utils import load_jsonl_pandas
from laysummarisation.utils import sentence_tokenize, preprocess
import re
from tqdm import tqdm

from process.greedy_rouge import process_entry, Arguments
from utils import set_seed


def get_binary_label_dataset(df: pd.DataFrame, tokenizer: BertTokenizerFast, device: str, check_consistency = False):
    """
    Get the dataset with the binary labels for each sentence in the article.

    :param df: The dataframe with the pseudo summary and the whole article
    :param tokenizer: The tokenizer to use
    :param device: The device to use
    :param check_consistency: Check if the pseudo summary is in the article (i.e., if preprocessing works correctly)
    :return: The dataset with the binary labels
    """
    # Store all article's sentences and their binary labels
    # (1 if the sentence is in the pseudo summary, 0 otherwise)
    article_sents = []
    article_sents_binary = []
    for i, row in tqdm(df.iterrows(), desc="Binary sentence extraction"):
        id_ = row['id']
        pseudo_summary = row['article']
        article_whole = row['article_whole']

        # Prepare for sentence tokenization
        pseudo_summary = preprocess(pseudo_summary)
        article_whole = preprocess(article_whole)

        # Tokenize sentences
        pseudo_summary_sents = sentence_tokenize(pseudo_summary)
        article_whole_sents = sentence_tokenize(article_whole)

        # Report sentence sizes
        print(f'ID: {id_}', end=' | ')
        print(f'Pseudo summary: {len(pseudo_summary_sents)}', end=' | ')
        print(f'Whole article: {len(article_whole_sents)}')

        if check_consistency:
            for sent in pseudo_summary_sents:
                if sent in article_whole_sents:
                    article_sents_binary.append(1)
                else:
                    article_sents_binary.append(0)
                    print("Sentence not found in article:", sent)
            print("--------")

        # Find the sentences that are in the pseudo summary
        for sent in article_whole_sents:
            article_sents.append(sent)
            if sent in pseudo_summary_sents:
                article_sents_binary.append(1)
            else:
                article_sents_binary.append(0)

    # Tokenize the sentences
    encoded = tokenizer(article_sents, truncation=True, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Create the dataset
    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.tensor(article_sents_binary).to(device)
    })

    return dataset


def main():
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # set_seed(conf.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # load Pseudo-summarising dataset
    root = "../../"
    # root = ""  # root directory of the project
    filename = "eLife"
    directory = "data/input/"
    # pseudo_summary = "lexrank"
    pseudo_summary = "rouge"
    dtype = "train"
    file_path = f'{root}{directory}{pseudo_summary}/{filename}_{dtype}.jsonl'
    pseudo_summary_dataset = load_jsonl_pandas(file_path, nrows=2)

    # load original dataset
    directory = "data/orig/"
    dtype = "train"
    file_path = f'{root}{directory}{dtype}/{filename}_{dtype}.jsonl'
    article_dataset = load_jsonl_pandas(file_path, nrows=2)

    # Merge the pseudo-summary (based on ROUGE) and the original article
    df = pseudo_summary_dataset.merge(article_dataset[['article', 'id']], on='id', how='inner', suffixes=('', '_whole'))

    # Store all article's sentences and their binary labels
    # (1 if the sentence is in the pseudo summary, 0 otherwise)
    article_sents = []
    article_sents_binary = []
    check_consistency = False  # Check if the pseudo summary is in the article (i.e., if preprocessing works correctly)
    output_path = f'{root}data/tmp/{filename}_{dtype}_sentences.csv'
    for i, row in tqdm(df.iterrows(), desc="Binary sentence extraction"):
        id_ = row['id']
        pseudo_summary = row['article']
        article_whole = row['article_whole']

        # Prepare for sentence tokenization
        pseudo_summary = preprocess(pseudo_summary)
        article_whole = preprocess(article_whole)

        # Tokenize sentences
        pseudo_summary_sents = sentence_tokenize(pseudo_summary)
        article_whole_sents = sentence_tokenize(article_whole)

        # Report sentence sizes
        print(f'ID: {id_}', end=' | ')
        print(f'Pseudo summary: {len(pseudo_summary_sents)}', end=' | ')
        print(f'Whole article: {len(article_whole_sents)}')

        if check_consistency:
            for sent in pseudo_summary_sents:
                if sent in article_whole_sents:
                    article_sents_binary.append(1)
                else:
                    article_sents_binary.append(0)
                    print("Sentence not found in article:", sent)
            print("--------")

        # Find the sentences that are in the pseudo summary
        for sent in article_whole_sents:
            article_sents.append(sent)
            if sent in pseudo_summary_sents:
                article_sents_binary.append(1)
            else:
                article_sents_binary.append(0)

        # Export the sentences and their labels to a csv file
        df = pd.DataFrame({'sentence': article_sents, 'label': article_sents_binary})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()
