import os

import torch
from datasets import Dataset
from transformers import (
    HfArgumentParser,
    LEDConfig,
    LEDForConditionalGeneration,
    LEDTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from laysummarisation.config import LFParserConfig
from laysummarisation.utils import (
    compute_metrics,
    load_jsonl_pandas,
    process_data_to_model_inputs,
    set_seed,
)


def train(conf: LFParserConfig):
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    set_seed(conf.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    # Naming and paths
    model_checkpoint = "yikuan8/Clinical-Longformer"
    model_name = model_checkpoint.split("/")[-1]

    # TODO: Add the config to the model
    lf_config = LEDConfig.from_pretrained(model_checkpoint)

    # Set Generation hyperparameters

    lf_config.attention_window = [
        conf.attention_window
    ] * lf_config.num_hidden_layers
    lf_config.num_beams = conf.nbeams
    lf_config.max_length = conf.max_encode
    lf_config.min_length = conf.min_encode
    lf_config.length_penalty = conf.length_penalty
    lf_config.early_stopping = conf.early_stopping
    lf_config.no_repeat_ngram_size = 3

    tokenizer = LEDTokenizer.from_pretrained(model_checkpoint)
    model = LEDForConditionalGeneration.from_pretrained(
        model_checkpoint, config=lf_config
    )

    assert isinstance(model, LEDForConditionalGeneration)

    model.to(device)

    model.train()

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir=conf.save_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=conf.logging_steps,
        warmup_steps=conf.warmup_steps,
        save_total_limit=2,
        gradient_accumulation_steps=conf.gradient_accum_steps,
        learning_rate=conf.lr,
        per_device_train_batch_size=conf.batch_size,
        per_device_eval_batch_size=conf.batch_size,
        num_train_epochs=conf.epochs,
        weight_decay=conf.weight_decay,
        run_name=model_name,
        report_to=["wandb"],
    )

    train_df = load_jsonl_pandas(conf.ftrain)
    eval_df = load_jsonl_pandas(conf.fvalid)

    train_dataset = Dataset.from_pandas(train_df).map(
        lambda x: process_data_to_model_inputs(
            x, tokenizer, conf.max_encode, conf.max_decode
        ),
        batched=True,
        batch_size=conf.batch_size,
        remove_columns=["article", "lay_summary"],
    )

    eval_dataset = Dataset.from_pandas(eval_df).map(
        lambda x: process_data_to_model_inputs(
            x, tokenizer, conf.max_encode, conf.max_decode
        ),
        batched=True,
        batch_size=conf.batch_size,
        remove_columns=["article", "lay_summary"],
    )

    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "global_attention_mask",
            "labels",
        ],
    )

    eval_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "global_attention_mask",
            "labels",
        ],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()


def main(conf):
    train(conf)


if __name__ == "__main__":
    parser = HfArgumentParser(LFParserConfig)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
