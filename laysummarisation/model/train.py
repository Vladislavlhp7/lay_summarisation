import os

import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    LEDConfig,
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from laysummarisation.config import LFParserConfig
from laysummarisation.utils import (
    compute_metrics,
    create_article_dataset_dict,
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

    # Training hyperparameters
    # lr = 3e-5  # from paper
    # batch_size = 1  # GPU does not have enough memory for batch_size > 1
    # max_input_length = 4096
    # max_output_length = 1024
    # pre_summarise = True
    # num_train_epochs = 3

    # Naming and paths
    model_checkpoint = "yikuan8/Clinical-Longformer"
    model_name = model_checkpoint.split("/")[-1]
    path_to_model = "../../Clinical-Longformer"

    # TODO: Add the config to the model
    lf_config = LEDConfig()

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = LEDForConditionalGeneration.from_pretrained(path_to_model, lf_config)
    assert isinstance(model, LEDForConditionalGeneration)

    # Set Generation hyperparameters
    model.config.num_beams = conf.nbeams
    model.config.max_length = conf.max_encode
    model.config.min_length = conf.min_encode
    model.config.length_penalty = conf.length_penalty
    model.config.early_stopping = conf.early_stopping
    model.config.no_repeat_ngram_size = 3
    model.to(device)

    model.train()

    # TODO: Refactor to use the preprocessed data
    article_dataset = create_article_dataset_dict(
        filename=conf.ftrain,
        batch_size=conf.batch_size,
        tokenizer=tokenizer,
        max_input_length=conf.max_encode,
        max_output_length=conf.max_decode,
        pre_summarise=conf.pre_summarise,
    )

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
        load_best_model_at_end=True,
        run_name=model_name,
        report_to=["wandb"],
    )

    # WARN: This does not work
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=article_dataset["train"],
        eval_dataset=article_dataset["val"],
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()


def main(conf):
    train(conf)


if __name__ == "__main__":
    parser = HfArgumentParser(LFParserConfig)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.nrows == 0:
        conf.nrows = None

    main(conf)
