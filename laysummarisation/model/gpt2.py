import gc
from dataclasses import dataclass, field

import pandas as pd
import torch

# from torch.utils.data import Dataset
from datasets import Dataset
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


@dataclass
class Arguments:
    """
    Arguments for model training
    """

    ftrain: str = field(
        metadata={"help": "Train file (.jsonl)"},
    )
    fvalid: str = field(
        metadata={"help": "Validation file (.jsonl)"},
    )
    corpus: str = field(
        metadata={"help": "The corpus name"},
    )
    save_dir: str = field(
        metadata={"help": "The directory to save the model"},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "The random seed for model and training initialization"
        },
    )
    model_checkpoint: str = field(
        default="yikuan8/Clinical-Longformer",
        metadata={"help": "The model checkpoint path or name."},
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "The device to use for training"},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "The temperature for GPT."},
    )
    max_encode: int = field(
        default=1024,
        metadata={"help": "The max token length for the encoder"},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "The learning rate"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "The batch size"},
    )
    epochs: int = field(
        default=1,
        metadata={"help": "The number of epochs"},
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "The number of steps between saving checkpoints"},
    )


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


def main(conf: Arguments):
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
            "max_length": conf.max_encode,
            "temperature": conf.temperature,
        }
    }
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"{conf.save_dir}/{conf.corpus}",
        overwrite_output_dir=True,
        num_train_epochs=conf.epochs,
        per_device_train_batch_size=conf.batch_size,
        save_steps=conf.save_steps,
        save_total_limit=2,
        evaluation_strategy="epoch",
        learning_rate=conf.lr,
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
        batch_size=conf.batch_size,
        remove_columns=["article", "lay_summary"],
    )

    eval_dataset = Dataset.from_pandas(eval_df).map(
        lambda x: build_inputs(x["article"], x["lay_summary"], tokenizer),
        batched=True,
        batch_size=conf.batch_size,
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

    print(train_dataset[0])
    print(eval_dataset[0])
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


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
