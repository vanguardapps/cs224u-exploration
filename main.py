from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from tqdm import tqdm
from utils import relative_path
import evaluate
import torch


# Example of using multilingual tokenizer for NLLB

# from transformers import NllbTokenizerFast

# tokenizer = NllbTokenizerFast.from_pretrained(
#     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
# )
# example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
# expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
# inputs = tokenizer(example_english_phrase, return_tensors="pt")
# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(expected_translation_french, return_tensors="pt")
# inputs["labels"] = labels["input_ids"]


def print_sample(model, tokenizer, tgt_lang):
    output_ids = model.generate(
        tokenizer("hello, how are you?", return_tensors="pt")["input_ids"],
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
    )
    with tokenizer.as_target_tokenizer():
        print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))


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


# TODO: See: https://stackoverflow.com/questions/67958756/how-do-i-check-if-a-tokenizer-model-is-already-saved
# for a description of how to check filepath and decide to use model_dir automatically without explicit
# parameter use_model_dir


def load_primary_components(
    model_name=None,
    model_dir=None,
    use_model_dir=False,
    max_sequence_length=None,
    dataset_filepath=None,
    metrics_list=None,
    token=True,
    tokenizer_kwargs={},
    model_kwargs={},
):
    dataset = get_split_dataset(dataset_filepath)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, **tokenizer_kwargs
    )
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(model_dir, token=token, **model_kwargs)
        if use_model_dir
        else AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=token, **model_kwargs
        )
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=max_sequence_length,
    )
    compute_metrics = partial(compute_metrics_generic, tokenizer, metrics_list)
    return dataset, tokenizer, model, data_collator, compute_metrics


def main():
    load_dotenv()  # Sets HF_TOKEN

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using CUDA.")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU.")

    eval_only_mode = False
    max_proc = 8
    max_sequence_length = 128
    model_dir = relative_path("model_output")
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    tokenization_batch_size = 500
    use_input_prefix = False
    use_model_dir = False

    dataset, tokenizer, model, data_collator, compute_metrics = load_primary_components(
        model_name="facebook/nllb-200-distilled-600M",
        model_dir=model_dir,
        max_sequence_length=max_sequence_length,
        dataset_filepath="de-en-europat.csv",
        metrics_list=["sacrebleu"],
        use_model_dir=use_model_dir,
        token=True,
        tokenizer_kwargs=dict(src_lang=src_lang, tgt_lang=tgt_lang),
    )

    tokenize_function = partial(
        tokenize_function_generic,
        tokenizer,
        max_sequence_length,
        "en",
        "de",
    )

    if use_input_prefix:
        add_input_prefix = partial(
            add_input_prefix_generic, "en", "translate english to german: "
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

    if eval_only_mode:
        print(
            evaluate_only(
                compute_metrics=compute_metrics,
                eval_dataset=tokenized_dataset["test"],
                model=model,
                max_new_tokens=max_sequence_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
    else:
        train_args = Seq2SeqTrainingArguments(
            adam_beta1=0.9,
            adam_beta2=0.999,
            evaluation_strategy="epoch",
            fp16=False if device == "cpu" else True,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,  # Set to True to improve memory utilization (though will slow training by 20%)
            learning_rate=5e-5,
            load_best_model_at_end=True,  # Always load the checkpoint that performed highest at the end
            logging_strategy="epoch",
            num_train_epochs=1,
            output_dir=model_dir,
            overwrite_output_dir=True,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=8,  # retest this limit now that fp16 is turned on
            predict_with_generate=True,  # Research this more
            push_to_hub=False,
            optim="adamw_torch",
            save_strategy="epoch",
            torch_compile=False,
            use_cpu=(
                True if device == "cpu" else False
            ),  # Set to False to automatically enable CUDA / mps device
            # warmup_steps=200,
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

        trainer.train(resume_from_checkpoint=True if use_model_dir else False)
        # tokenizer.save_pretrained(model_dir)
        trainer.save_model(model_dir)


if __name__ == "__main__":
    main()
