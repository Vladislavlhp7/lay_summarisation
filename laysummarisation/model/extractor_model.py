import os
from dataclasses import dataclass, field

import torch
import wandb
from scipy.special import softmax
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          HfArgumentParser, Trainer, TrainingArguments, BertConfig)

from laysummarisation.utils import (compute_binary_metrics, load_binary_data,
                                    set_seed, sentence_tokenize, preprocess)


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


def generate_summary(model_path: str, article: str, max_length: int = 512, device: str = "cpu", *args):
    """
    Generate summary from the BERT model

    Args:
        model_path (str): The path to the model.
        article (str): The article to summarise.
        max_length (int): The maximum number of tokens to generate.
        device (str): The device to use.
        *args: The input arguments to the Trainer.

    Returns:
        summary (str): The generated summary.
    """
    # Load model
    model_dir = os.path.dirname(model_path)
    output_model_file = f"{model_dir}/pytorch_model.bin"
    output_config_file = f"{model_dir}/config.json"
    config = BertConfig.from_json_file(output_config_file)
    model = BertForSequenceClassification(config)
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    # Prepare article for sentence tokenization
    article_cleaned = preprocess(article)
    # Segment article into sentences
    article_segmented = sentence_tokenize(article)
    # Tokenize article sentences
    article_tokenized = tokenizer(article_segmented, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

    # Generate summarising sentence probabilities
    trainer = Trainer(model=model, args=args)
    model.eval()
    with torch.no_grad():
        outputs = trainer.predict(article_tokenized)
        predictions = torch.tensor(softmax(outputs.predictions, axis=1))
        # TODO: Take Top-K sentences


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
