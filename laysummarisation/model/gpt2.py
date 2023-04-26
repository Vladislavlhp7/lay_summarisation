from dataclasses import dataclass, field
import gc
import pandas as pd
import torch
# import wandb
from datasets import load_metric
from rouge import Rouge
from scipy.special import softmax
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (GPT2Config, GPT2LMHeadModel,
                          GPT2Tokenizer, HfArgumentParser, Trainer,
                          TrainingArguments)


torch.empty_cache()

gc.collect()


def build_inputs(text: str, summary: str, tokenizer: GPT2Tokenizer, max_length: int = 1024) -> dict:
    
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
    target_ids = tokenizer.encode(summary, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
    
    return {"input_ids": input_ids, "labels": target_ids}


def compute_metrics(eval_pred, tokenizer: GPT2Tokenizer) -> dict:

    predictions, labels = eval_pred

    pred_ids = np.argmax(predictions, axis=-1)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = Rouge()
    scores = rouge.get_scores(pred_str, labels_str, avg=True)

    accuracy = sum([1 if p == l else 0 for p, l in zip(pred_str, labels_str)]) / len(labels_str)

    results = {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"],
        "accuracy": accuracy,
    }

    return results


def main():
    # print("CUDA available:" + str(torch.cuda.is_available()))
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if device == "cuda":
    #     torch.cuda.empty_cache()

    # # Load files
    # print("Loading files...")

    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    config.task_specific_params = {
        'text-generation': {'do_sample': True, 'max_length': 256, 'temperature': 0.7}
    }
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    # model.to(device)

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

    train_df = pd.read_json("./data/input/rouge/eLife_train.jsonl", lines=True).head(100)
    eval_df = pd.read_json("./data/input/rouge/eLife_val.jsonl", lines=True).head(10)

    # Create the train and evaluation datasets
    train_dataset = [build_inputs(row['article'], row['lay_summary'], tokenizer) for _, row in train_df.iterrows()]
    eval_dataset = [build_inputs(row['article'], row['lay_summary'], tokenizer) for _, row in eval_df.iterrows()]

    # Initialize the trainer with the model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained("./lay_summary_model")
    tokenizer.save_pretrained("./lay_summary_model")

if __name__ == "__main__":
    main()

