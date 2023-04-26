import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

# from torch.utils.data import Dataset
from datasets import Dataset, load_metric
from rouge import Rouge
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from laysummarisation.utils import compute_metrics

gc.collect()


def build_inputs(
    text: str, summary: str, tokenizer: GPT2Tokenizer, max_length: int = 1024
) -> dict:
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    target_ids = tokenizer.encode(
        summary,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    return {"input_ids": input_ids, "labels": target_ids}


# class LaySummarizationDataset(Dataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         return {
#             "input_ids": item["input_ids"].squeeze(),
#             "labels": item["labels"].squeeze(),
#         }
#


def main():
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # # Load files
    # print("Loading files...")

    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    config.task_specific_params = {
        "text-generation": {
            "do_sample": True,
            "max_length": 256,
            "temperature": 0.7,
        }
    }
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./lay_summary_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
    )

    train_df = pd.read_json(
        "./data/input/rouge/eLife_train.jsonl", lines=True
    ).head(10)
    eval_df = pd.read_json(
        "./data/input/rouge/eLife_val.jsonl", lines=True
    ).head(10)

    train_dataset = Dataset.from_pandas(train_df).map(
        lambda x: build_inputs(x["article"], x["lay_summary"], tokenizer),
        batched=True,
        batch_size=1,
        remove_columns=["article", "lay_summary"],
    )

    eval_dataset = Dataset.from_pandas(eval_df).map(
        lambda x: build_inputs(x["article"], x["lay_summary"], tokenizer),
        batched=True,
        batch_size=1,
        remove_columns=["article", "lay_summary"],
    )

    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "labels",
        ],
    )

    eval_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "labels",
        ],
    )

    # Create the train and evaluation datasets
    # train_dataset = LaySummarizationDataset(
    #     [
    #         build_inputs(row["article"], row["lay_summary"], tokenizer)
    #         for _, row in train_df.iterrows()
    #     ],
    #     tokenizer,
    # )
    # eval_dataset = LaySummarizationDataset(
    #     [
    #         build_inputs(row["article"], row["lay_summary"], tokenizer)
    #         for _, row in eval_df.iterrows()
    #     ],
    #     tokenizer,
    # )

    # Initialize the trainer with the model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained("./lay_summary_model")
    tokenizer.save_pretrained("./lay_summary_model")


if __name__ == "__main__":
    main()
