import json
import inspect
import numpy as np

from datasets import Dataset, Features, Value, Sequence, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate


# --- 1) CONFIGURATION ---

MODEL_CHECKPOINT = "distilbert-base-uncased" 
NER_DATA_FILE = "ner_training_data.json"
OUTPUT_MODEL_DIR = "surgical_ner_model" 


# --- 2) NER SCHEMA (IOB2 style) ---

label_names = ["INSTRUMENT", "ANATOMY", "ACTION", "OBSERVATION"]
tags = ClassLabel(
    names=["O"] + [f"B-{l}" for l in label_names] + [f"I-{l}" for l in label_names]
)


# --- 3) DATA LOADING & PREPROCESSING ---

def load_and_prepare_dataset(json_path, tokenizer):
    """Load JSON and convert to HF Dataset with pre-tokenized 'tokens' and tag ids."""
    print(f"Loading dataset from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset_dict = {"tokens": [], "ner_tags": []}

    for example in data:
        # Tokenize full text into subword tokens 
        tokenized_for_words = tokenizer(example["text"], add_special_tokens=False)
        words = tokenizer.convert_ids_to_tokens(tokenized_for_words["input_ids"])
        dataset_dict["tokens"].append(words)

        # Start as all 'O'
        word_tags = ["O"] * len(words)

        # Simplified alignment: try to match tokenized entity span to token list
        for label_info in example.get("labels", []):
            start, end = label_info["start"], label_info["end"]
            entity_label = label_info["label"]
            entity_tokens = tokenizer.tokenize(example["text"][start:end])

            # Find matching window
            for i in range(len(words) - len(entity_tokens) + 1):
                if words[i:i + len(entity_tokens)] == entity_tokens:
                    word_tags[i] = f"B-{entity_label}"
                    for j in range(1, len(entity_tokens)):
                        word_tags[i + j] = f"I-{entity_label}"
                    break

        dataset_dict["ner_tags"].append([tags.str2int(t) for t in word_tags])

    features = Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(feature=tags)})
    dataset = Dataset.from_dict(dataset_dict, features=features)
    print("Dataset loaded and prepared.")
    return dataset


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize pre-split words and align labels to subword ids."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False 
    )

    all_labels = examples["ner_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids, prev_word_idx = [], None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)             
            elif word_idx != prev_word_idx:
                label_ids.append(labels[word_idx]) 
            else:
                label_ids.append(-100)            
            prev_word_idx = word_idx

        new_labels.append(label_ids)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# --- 4) METRICS ---

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """Version-robust: supports tuple or EvalPrediction."""
    try:
        predictions, labels = p
    except Exception:
        predictions, labels = p.predictions, p.label_ids

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [tags.int2str(pp) for pp, ll in zip(pred, lab) if ll != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [tags.int2str(ll) for pp, ll in zip(pred, lab) if ll != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "f1": results.get("overall_f1", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
    }


# --- 5) TRAINING ARGUMENTS ---

def build_training_args():
    """
    Build TrainingArguments compatible with the installed transformers version by
    inspecting the __init__ signature and only passing supported kwargs.
    Crucially, only set `load_best_model_at_end=True` when BOTH
    `evaluation_strategy` and `save_strategy` are supported so strategies can match.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())

    desired = {
        "output_dir": OUTPUT_MODEL_DIR,
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "weight_decay": 0.01,
        "logging_steps": 500,
        "eval_steps": 500,
        "save_steps": 500,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
    }

    kwargs = {}

    # Basics
    for k in ["output_dir", "learning_rate", "num_train_epochs", "weight_decay", "logging_steps"]:
        if k in valid:
            kwargs[k] = desired[k]

    # Batch sizes 
    if "per_device_train_batch_size" in valid:
        kwargs["per_device_train_batch_size"] = desired["per_device_train_batch_size"]
        if "per_device_eval_batch_size" in valid:
            kwargs["per_device_eval_batch_size"] = desired["per_device_eval_batch_size"]
    else:
        if "per_gpu_train_batch_size" in valid:
            kwargs["per_gpu_train_batch_size"] = desired["per_device_train_batch_size"]
        if "per_gpu_eval_batch_size" in valid:
            kwargs["per_gpu_eval_batch_size"] = desired["per_device_eval_batch_size"]

    # Save/eval cadence if available
    if "eval_steps" in valid:
        kwargs["eval_steps"] = desired["eval_steps"]
    if "save_steps" in valid:
        kwargs["save_steps"] = desired["save_steps"]

    # Strategy coordination
    eval_supported = "evaluation_strategy" in valid
    save_supported = "save_strategy" in valid
    lbme_supported = "load_best_model_at_end" in valid

    if eval_supported:
        kwargs["evaluation_strategy"] = "steps"
    else:
        # Some older versions have 'do_eval' or 'evaluate_during_training'
        if "do_eval" in valid:
            kwargs["do_eval"] = True
        if "evaluate_during_training" in valid:
            kwargs["evaluate_during_training"] = True

    if save_supported:
        kwargs["save_strategy"] = "steps"

    # Only enable load_best_model_at_end when both strategies exist so they can match
    if lbme_supported and eval_supported and save_supported:
        kwargs["load_best_model_at_end"] = True

    print("TrainingArguments supported keys being used:", sorted(kwargs.keys()))
    return TrainingArguments(**kwargs)

# --- 6) MAIN ---

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    dataset = load_and_prepare_dataset(NER_DATA_FILE, tokenizer)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(tags.names)}
    label2id = {label: i for i, label in enumerate(tags.names)}
    num_labels = len(tags.names)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting model fine-tuning...")
    trainer.train()

    # Final evaluation 
    try:
        print("Final evaluation:")
        metrics = trainer.evaluate()
        print(metrics)
    except Exception as e:
        print(f"Evaluation skipped (not supported in this transformers version): {e}")

    trainer.save_model(OUTPUT_MODEL_DIR)

    print("\n-----------------------------------------")
    print("NER model training complete!")
    print(f"The fine-tuned model has been saved to the '{OUTPUT_MODEL_DIR}' directory.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
