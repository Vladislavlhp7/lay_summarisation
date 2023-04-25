import pandas as pd
import torch
from laysummarisation.utils import compute_metrics

from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    TextDataset, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def process_data_to_model_inputs(batch):
    """
    Preprocesses data into model inputs.
    
    Args:
        batch: A dictionary containing 'article' and 'lay_summary'.
    
    Returns:
        A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    inputs = tokenizer(batch['article'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    outputs = tokenizer(batch['lay_summary'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

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
    train_df = pd.read_json("../../laySummarisation/data/input/rouge/eLife_train.jsonl", lines=True)
    eval_df = pd.read_json("../../laySummarisation/data/input/rouge/eLife_val.jsonl", lines=True)

    # Load Tokenizer and Model
    model_name = "../clinical-t5/1.0.0/Clinical-T5-Sci/"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="../laySummarisation/data/input/rouge/eLife_train.jsonl",
        column_names=["article", "lay_summary"],
        block_size=512,
    )
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="../laySummarisation/data/input/rouge/eLife_val.jsonl",
        column_names=["article", "lay_summary"],
        block_size=512,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
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
        compute_metrics=compute_metrics, # The `compute_metrics` function needs to be defined before calling this.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model("fine_tuned_Clinical-T5-Sci")


if __name__ == "__main__":
    main()