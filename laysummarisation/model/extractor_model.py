import os
from dataclasses import dataclass, field
from typing import List

import torch
import wandb
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          HfArgumentParser, Trainer, TrainingArguments, BertConfig)

from inference.extractor_model import generate_summary
from laysummarisation.utils import (compute_binary_metrics, load_binary_data,
                                    set_seed)


@dataclass
class Arguments:
    """
    Arguments
    """

    fname: str = field(
        metadata={"help": "The input binary label dataset file path"},
    )
    lr: float = field(
        default=2e-5,
        metadata={"help": "The learning rate."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )


def load_extractor_model(model_path: str, device: str = "cpu"):
    """
    Load the extractor model

    Args:
        model_path (str): The path to the model.
        device (str): The device to use.

    Returns:
        model (BertForSequenceClassification): The model.
        tokenizer (BertTokenizerFast): The tokenizer.
    """
    # Load model
    model_dir = os.path.dirname(model_path)
    output_model_file = f"{model_dir}/pytorch_model.bin"
    output_config_file = f"{model_dir}/config.json"
    config = BertConfig.from_json_file(output_config_file)
    model = BertForSequenceClassification(config)
    model.to(device)
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    # Load Tokenizer
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
    except OSError:
        model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer


def generate_summaries(model_path: str, articles: List[str], max_length: int = 512, top_k: int = 25,
                       device: str = "cpu", args: TrainingArguments = None):
    """
    Generate summaries from the BERT model

    Args:
        model_path (str): The path to the model.
        articles (List[str]): The articles to summarise.
        max_length (int): The maximum number of tokens to generate.
        top_k (int): The number of sentences to select.
        device (str): The device to use.
        args (TrainingArguments): The training arguments.

    Returns:
        summaries (List[str]): The generated summaries.
    """
    model, tokenizer = load_extractor_model(model_path, device)
    summaries = []
    for article in articles:
        summary = generate_summary(model=model, tokenizer=tokenizer, article=article,
                                   max_length=max_length, top_k=top_k, args=args)
        summaries.append(summary)
    return summaries


def main(conf: Arguments):
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load files
    print("Loading files...")

    set_seed(conf.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    assert isinstance(model, BertForSequenceClassification)
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_dataset, val_dataset, test_dataset = load_binary_data(
        fname=conf.fname, tokenizer=tokenizer, max_length=128
    )

    args = TrainingArguments(
        output_dir="../tmp/",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=conf.lr,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=conf.seed,
        logging_steps=7000,
        save_steps=7000,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_binary_metrics,
    )

    trainer.train()

    model.eval()
    with torch.no_grad():
        prediction_obj = trainer.predict(test_dataset)
        metrics = prediction_obj.metrics
        assert metrics is not None
        metrics = {f"test/{k}": v for k, v in metrics.items()}
        print(metrics)
        wandb.log(metrics)
    trainer.save_model(f'{model_name}_{conf.lr}')


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.fname is None:
        conf.fname = "../../data/tmp/extractive/rouge/PLOS_train.csv"
    main(conf)
