# from longformer_helper import *
import os

import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments)

from laysummarisation.config import LFParserConfig
from laysummarisation.utils import (compute_metrics,
                                    create_article_dataset_dict, set_seed)


def train(config):
    if config.device == "gpu":
        torch.cuda.empty_cache()
    set_seed(config.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    # lr = 3e-5  # from paper
    # batch_size = 1  # GPU does not have enough memory for batch_size > 1
    # max_input_length = 4096
    # max_output_length = 1024

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    path_to_model = "../../Clinical-Longformer"
    path_to_model = os.path.join(dir_path, path_to_model)
    model = AutoModelForMaskedLM.from_pretrained(path_to_model)
    model.to(config.device)

    model_name = config.model_checkpoint.split("/")[-1]

    article_dataset = create_article_dataset_dict(
        config.ftrain,
        config.batch_size,
        tokenizer,
        config.max_encode,
        config.max_decode,
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric,
        run_name=model_name,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=article_dataset["train"],
        eval_dataset=article_dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():
    parser = HfArgumentParser(LFParserConfig)
    train(parser.parse_args_into_dataclasses())
    exit()


if __name__ == "__main__":
    main()
