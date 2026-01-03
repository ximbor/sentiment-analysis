import os
import numpy as np
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

def train():
    model_name = "ximbor/sentiment-monitor"

    learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
    train_epochs = int(os.getenv("TRAIN_EPOCHS", "3"))
    train_batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "8"))

    train_size = int(os.getenv("DATASET_TRAIN_SIZE", "0"))
    test_size = int(os.getenv("DATASET_TEST_SIZE", "0"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("tweet_eval", "sentiment")

    if train_size > 0:
        dataset["train"] = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
    if test_size > 0:
        dataset["test"] = dataset["test"].select(range(min(test_size, len(dataset["test"]))))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print("Evaluating current production model...")
    current_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    baseline_trainer = Trainer(
        model=current_model,
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics
    )
    baseline_results = baseline_trainer.evaluate()
    baseline_f1 = baseline_results["eval_f1_macro"]
    print(f"Current model F1-Macro: {baseline_f1:.4f}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=train_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics
    )

    print("Training new model...")
    trainer.train()

    new_results = trainer.evaluate()
    new_f1 = new_results["eval_f1_macro"]
    print(f"New model F1-Macro: {new_f1:.4f}")

    if new_f1 > baseline_f1:
        print(f"New model ({new_f1:.4f}) is better than current ({baseline_f1:.4f}).")
        print("Saving model...")
        trainer.save_model("./tmp_model")
        tokenizer.save_pretrained("./tmp_model")
        print("Model saved: './tmp_model'")
    else:
        print(f"New model ({new_f1:.4f}) did not outperform current ({baseline_f1:.4f}).")
        print("Skipping...")
        sys.exit(1)


if __name__ == "__main__":
    train()