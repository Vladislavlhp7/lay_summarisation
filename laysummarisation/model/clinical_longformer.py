from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from longformer_helper import *


def main():
    print('CUDA available:' + str(torch.cuda.is_available()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    set_seed(42)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    lr = 3e-5  # from paper
    batch_size = 32
    max_input_length = 4096
    max_output_length = 1024

    model_checkpoint = "yikuan8/Clinical-Longformer"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    path_to_model = "../../Clinical-Longformer"
    model = AutoModelForMaskedLM.from_pretrained(path_to_model)
    model.to(device)
    
    num_train_epochs = 8
    model_name = model_checkpoint.split("/")[-1]

    filename = "eLife"
    directory = "../../data/task1_development/"
    article_dataset = create_article_dataset_dict(filename, directory, batch_size, tokenizer, max_input_length)

    args = TrainingArguments(
        output_dir='../../tmp/',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='rouge2_f',
        run_name=model_name,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=article_dataset['train'],
        eval_dataset=article_dataset['val'],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
