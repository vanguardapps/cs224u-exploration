from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
from functools import partial
from nlp_utils import relative_path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_split_dataset(dataset_filepath):
    dataset_all = load_dataset("csv", data_files=relative_path(dataset_filepath))
    train_testvalidate = dataset_all["train"].train_test_split(
        test_size=0.3, shuffle=True
    )
    test_validate = train_testvalidate["test"].train_test_split(
        test_size=0.5, shuffle=True
    )
    return DatasetDict(
        {
            "train": train_testvalidate["train"],
            "test": test_validate["train"],
            "validate": test_validate["test"],
        }
    )


def compute_metrics_generic(tokenizer, metrics_list, eval_preds):
    # print("compute metrics bypassed due to bug")
    metric = evaluate.load(*metrics_list)

    predictions, references = eval_preds
    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = tokenizer.batch_decode(
        references, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = [[reference] for reference in references]
    return metric.compute(predictions=predictions, references=references)


def load_primary_components(
    model_name, max_sequence_length, dataset_filepath, metrics_list
):
    dataset = get_split_dataset(dataset_filepath)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=max_sequence_length,
    )
    compute_metrics = partial(compute_metrics_generic, tokenizer, metrics_list)
    return dataset, tokenizer, model, data_collator, compute_metrics


def tokenize_function_generic(
    tokenizer, max_sequence_length, input_property, labels_property, batch
):
    input_feature = tokenizer(
        batch[input_property],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
    )
    labels = tokenizer(
        batch[labels_property],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
    )
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": labels["input_ids"],
        # "token_type_id" is sometimes also included here, usually with BERT-like models
    }


def add_input_prefix_generic(input_property, prefix, batch):
    batch[input_property] = [
        prefix + example_input for example_input in batch[input_property]
    ]
    return batch


def evaluate_only(compute_metrics, eval_dataset, model, max_new_tokens, pad_token_id):
    loader = DataLoader(eval_dataset, batch_size=100)
    predictions = None
    references = None
    placeholder_tensor = [torch.LongTensor(max_new_tokens)]

    print("Predicting over eval dataset...")
    for batch in tqdm(loader):
        batch_predictions = model.generate(
            batch["input_ids"], max_new_tokens=max_new_tokens
        ).tolist()
        batch_predictions = pad_sequence(
            [
                torch.LongTensor(example)
                for example in (placeholder_tensor + batch_predictions)
            ],
            padding_value=pad_token_id,
            batch_first=True,
        )
        batch_predictions = batch_predictions[1:]

        if predictions is None:
            predictions = batch_predictions
            references = batch["labels"]
        else:
            predictions = torch.cat(
                (
                    predictions,
                    batch_predictions,
                ),
                0,
            )
            references = torch.cat((references, batch["labels"]), 0)

    eval_preds = (predictions, references)
    return compute_metrics(eval_preds)


def main():
    local_cpu = False

    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
    else:
        print("CUDA not available. Using CPU.")

    max_sequence_length = 128
    tokenization_batch_size = 500
    max_proc = 8

    dataset, tokenizer, model, data_collator, compute_metrics = load_primary_components(
        model_name="google/mt5-small",
        max_sequence_length=max_sequence_length,
        dataset_filepath="data/english_to_spanish.csv",
        metrics_list=["sacrebleu"],
    )

    tokenize_function = partial(
        tokenize_function_generic, tokenizer, max_sequence_length, "english", "spanish"
    )

    add_input_prefix = partial(
        add_input_prefix_generic, "english", "translate english to spanish: "
    )

    dataset = dataset.map(
        add_input_prefix,
        batched=True,
        batch_size=tokenization_batch_size,
        num_proc=max_proc,
    )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenization_batch_size,
        num_proc=max_proc,
    )

    tokenized_dataset.set_format(type="torch")

    # Uncomment to evaluate dataset against current model only
    # print(
    #     evaluate_only(
    #         compute_metrics=compute_metrics,
    #         eval_dataset=tokenized_dataset["test"],
    #         model=model,
    #         max_new_tokens=max_sequence_length,
    #         pad_token_id=tokenizer.pad_token_id,
    #     )
    # )

    train_args = Seq2SeqTrainingArguments(
        use_cpu=True
        if local_cpu
        else False,  # Set to False to automatically enable CUDA / mps device
        per_device_train_batch_size=8,  # retest this limit now that fp16 is turned on
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False if local_cpu else True,
        output_dir="model_output",
        overwrite_output_dir=True,
        push_to_hub=False,
        learning_rate=5e-5,
        logging_strategy="epoch",
        optim="adamw_torch",
        # warmup_steps=200,
        predict_with_generate=True,  # Research this more
        adam_beta1=0.9,
        adam_beta2=0.999,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # Set to True to improve memory utilization (though will slow training by 20%)
        torch_compile=False,
    )

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()
