import os
from dataclasses import dataclass, field

import torch
import wandb
from datasets import Dataset
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          HfArgumentParser, Trainer, TrainingArguments)

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
    model = BertForSequenceClassification.from_pretrained(model_name)
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
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=conf.seed,
        logging_steps=100,
        save_steps=100,
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
    trainer.save_model(model_name)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.fname is None:
        conf.fname = "../../data/tmp/extractive/rouge/PLOS_train.csv"
    main(conf)
