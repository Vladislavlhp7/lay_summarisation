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

import wandb
from laysummarisation.utils import compute_metrics


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
    batch, tokenizer: GPT2Tokenizer, max_length: int = 512
) -> dict:
    batch["input_ids"] = tokenizer.encode(
        batch["article"],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    batch["labels"] = tokenizer.encode(
        batch["lay_summary"],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    return batch


def main(conf: Arguments):
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # # Load files
    # print("Loading files...")

    config = GPT2Config.from_pretrained(conf.model_checkpoint)
    config.task_specific_params = {
        "text-generation": {
            "do_sample": True,
            "max_length": conf.max_encode,
            "temperature": conf.temperature,
        }
    }
    tokenizer = GPT2Tokenizer.from_pretrained(conf.model_checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    model = GPT2LMHeadModel.from_pretrained(
        conf.model_checkpoint,
        config=config,
    )

    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"{conf.save_dir}/{conf.corpus}",
        overwrite_output_dir=True,
        num_train_epochs=conf.epochs,
        per_device_train_batch_size=conf.batch_size,
        per_device_eval_batch_size=conf.batch_size,
        save_steps=conf.save_steps,
        save_total_limit=2,
        evaluation_strategy="epoch",
        report_to=["wandb"],
        learning_rate=conf.lr,
        fp16=True,
        fp16_full_eval=True,
        # gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
    )

    train_df = pd.read_json(conf.ftrain, lines=True).head(100)
    eval_df = pd.read_json(conf.fvalid, lines=True).head(100)

    train_dataset = Dataset.from_pandas(train_df).map(
        lambda x: build_inputs(x, tokenizer, max_length=conf.max_encode),
        remove_columns=["article", "lay_summary"],
    )

    eval_dataset = Dataset.from_pandas(eval_df).map(
        lambda x: build_inputs(x, tokenizer, max_length=conf.max_encode),
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

    # Initialize the trainer with the model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, batched=True),
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
