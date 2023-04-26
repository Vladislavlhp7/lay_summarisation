import os

import torch
from datasets import Dataset
from transformers import (HfArgumentParser, LEDConfig,
                          LEDForConditionalGeneration, LEDTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from laysummarisation.config import LFParserConfig
from laysummarisation.utils import (compute_metrics, load_jsonl_pandas,
                                    process_data_to_model_inputs, set_seed)


def generate_summary(model, tokenizer, article: str, max_length: int = 512):
    """
    Generate summary from the Clinical Longformer model
    Args:
        model: The model.
        tokenizer: The tokenizer.
        article: The article to summarise.
        max_length: The maximum number of tokens to generate.
        args: The input arguments to the Trainer.
    Returns:
        summary (str): The generated summary.
    """
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(
            article,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        input_ids = input_ids.to(model.device)

        output = model.generate(
            input_ids, max_length=max_length, num_return_sequences=1
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


def load_longformer_model(model_path: str, device: str = "cpu"):
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
    config = LEDConfig.from_json_file(output_config_file)
    model = LEDForConditionalGeneration.from_pretrained(config)
    model.to(device)
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    # Load Tokenizer
    try:
        tokenizer = LEDTokenizer.from_pretrained(model_path)
    except OSError:
        model_name = "yikuan8/Clinical-Longformer"
        tokenizer = LEDTokenizer.from_pretrained(model_name)
    return model, tokenizer


def train(conf: LFParserConfig):
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    set_seed(conf.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    # Naming and paths
    model_name = conf.model.split("/")[-1]

    if conf.checkpoint is not None:
        conf.model = conf.checkpoint

    # TODO: Add the config to the model
    lf_config = LEDConfig.from_pretrained(conf.model)

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

    tokenizer = LEDTokenizer.from_pretrained(conf.model)
    model = LEDForConditionalGeneration.from_pretrained(
        conf.model, config=lf_config
    )

    assert isinstance(model, LEDForConditionalGeneration)

    model.to(device)

    model.train()

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir=conf.save_dir,
        evaluation_strategy=conf.eval_strategy,
        logging_strategy=conf.logging_strategy,
        seed=conf.seed,
        save_strategy=conf.save_strategy,
        logging_steps=conf.logging_steps,
        warmup_steps=conf.warmup_steps,
        save_total_limit=2,
        gradient_accumulation_steps=conf.gradient_accum_steps,
        learning_rate=conf.lr,
        per_device_train_batch_size=conf.batch_size,
        per_device_eval_batch_size=conf.batch_size,
        num_train_epochs=conf.epochs,
        weight_decay=conf.weight_decay,
        run_name=f"{model_name}_{conf.corpus}",
        report_to=["wandb"],
        eval_steps=conf.eval_steps,
        fp16=conf.fp16,
        fp16_full_eval=conf.fp16_full_eval,
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

    trainer.train(resume_from_checkpoint=conf.checkpoint is not None)


def main(conf):
    train(conf)


if __name__ == "__main__":
    parser = HfArgumentParser(LFParserConfig)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
