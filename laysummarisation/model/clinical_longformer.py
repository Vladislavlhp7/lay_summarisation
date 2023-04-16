import os

import torch
from transformers import (AutoTokenizer)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, LEDForConditionalGeneration

from laysummarisation.utils import (compute_metrics,
                                    create_article_dataset_dict, set_seed)


def main():
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    set_seed(42)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    # Training hyperparameters
    lr = 3e-5  # from paper
    batch_size = 1  # GPU does not have enough memory for batch_size > 1
    max_input_length = 4096
    max_output_length = 1024
    pre_summarise = True
    num_train_epochs = 3

    # Naming and paths
    model_checkpoint = "yikuan8/Clinical-Longformer"
    model_name = model_checkpoint.split("/")[-1]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_model = "../../Clinical-Longformer"

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = LEDForConditionalGeneration.from_pretrained(path_to_model)
    # Set Generation hyperparameters
    model.config.num_beams = 4
    model.config.max_length = 512
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.to(device)

    filename = "eLife"
    directory = "../../data/task1_development/"
    directory = os.path.join(dir_path, directory)
    article_dataset = create_article_dataset_dict(filename=filename, directory=directory, batch_size=batch_size,
                                                  tokenizer=tokenizer, max_input_length=max_input_length,
                                                  max_output_length=max_output_length, pre_summarise=pre_summarise)

    output_dir = "../../tmp/"
    output_dir = os.path.join(dir_path, output_dir)
    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=250,
        warmup_steps=1500,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        run_name=model_name,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=article_dataset["train"],
        eval_dataset=article_dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
