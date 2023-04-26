import pandas as pd
import torch
from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer, TextDataset)

import wandb
from laysummarisation.utils import compute_metrics


def process_data_to_model_inputs(batch, tokenizer):
    """
    Preprocesses data into model inputs.

    Args:
        batch: A dictionary containing 'article' and 'lay_summary'.
        tokenizer: The tokenizer.

    Returns:
        A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    outputs = tokenizer(
        batch["lay_summary"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs["input_ids"] = inputs["input_ids"].squeeze()
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    outputs["input_ids"] = outputs["input_ids"].squeeze()
    outputs["attention_mask"] = outputs["attention_mask"].squeeze()
    labels = outputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }


def main():
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load files
    print("Loading files...")

    # Load files
    # train_df = pd.read_json("../../laySummarisation/data/input/rouge/eLife_train.jsonl", lines=True)
    # eval_df = pd.read_json("../../laySummarisation/data/input/rouge/eLife_val.jsonl", lines=True)

    # Load Tokenizer and Model
    model_name = "./weights/Clinical-T5-Sci/"
    tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)


    # Load files
    train_df = pd.read_json("./data/input/rouge/eLife_train.jsonl", lines=True)
    eval_df = pd.read_json("./data/input/rouge/eLife_val.jsonl", lines=True)

    # Create Datasets from pandas DataFrames
    train_dataset = Dataset.from_pandas(train_df).map(
        lambda x: process_data_to_model_inputs(x, tokenizer),
        batched=True,
        remove_columns=["article", "lay_summary"],
    )
    eval_dataset = Dataset.from_pandas(eval_df).map(
        lambda x: process_data_to_model_inputs(x, tokenizer),
        batched=True,
        remove_columns=["article", "lay_summary"],
    )


    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp/t5",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        num_train_epochs=3,
        logging_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=1,
        fp16=True,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge2_fmeasure",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,  # The `compute_metrics` function needs to be defined before calling this.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("fine_tuned_Clinical-T5-Sci")


if __name__ == "__main__":
    main()
