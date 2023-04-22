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
    print(file_path)
    pseudo_summary_dataset = load_jsonl_pandas(file_path, nrows=1)

    # load original dataset
    directory = "data/orig/"
    dtype = "train"
    file_path = f'{root}{directory}{dtype}/{filename}_{dtype}.jsonl'
    print(file_path)
    article_dataset = load_jsonl_pandas(file_path, nrows=1)

    # 1. Preprocess the data
    # confs = Arguments(nsent=1, fname=file_path, output='../data/output/rouge_cleaned.jsonl')
    # article_dataset["rouge_cleaned"] = article_dataset.apply(lambda x: process_entry(x, confs), axis=1)
    # assert that sentences from the pseudo-summary are in the original article
    # for i, row in tqdm(article_dataset.iterrows(), desc="Binary sentence extraction"):
    #     id_ = row['id']
    #     rouge_cleaned = row['rouge_cleaned']
    #     article = row['article']
    #
    #     # Prepare for sentence tokenization
    #     rouge_cleaned = preprocess(rouge_cleaned)
    #     article = preprocess(article)
    #
    #     # Tokenize sentences
    #     rouge_cleaned_sents = sentence_tokenize(rouge_cleaned)
    #     print(rouge_cleaned_sents)
    #     article_sents = sentence_tokenize(article)
    #
    #     # Report sentence sizes
    #     print(f'ID: {id_}', end=' | ')
    #     print(f'Pseudo summary: {len(rouge_cleaned_sents)}', end=' | ')
    #     print(f'Article: {len(article_sents)}', end=' | ')
    #     print(f'Intersection: {len(set(rouge_cleaned_sents).intersection(set(article_sents)))}')

    df = pseudo_summary_dataset.merge(article_dataset[['article', 'id']], on='id', how='inner', suffixes=('', '_whole'))

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

        # Find the sentences that are in the pseudo summary
        pseudo_summary_sents_binary = []
        for sent in pseudo_summary_sents:
            if sent in article_whole_sents:
                pseudo_summary_sents_binary.append(1)
            else:
                pseudo_summary_sents_binary.append(0)
                print("Sentence not found in article:", sent)
        print("--------")
        for s in pseudo_summary_sents:
            print(s)
        # Report the number of sentences that are in the pseudo summary
        print(f'Pseudo summary sentences: {sum(pseudo_summary_sents_binary)}')

        break


if __name__ == "__main__":
    main()
