from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

def train():
    MODEL_NAME = "ximbor/sentiment-monitor"
    LEARNING_RATE = 2e-5
    TRAIN_EPOCHS = 3
    TRAIN_BATCH_SIZE = 8

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    dataset = load_dataset("tweet_eval", "sentiment")
    
    # Limit the dataset size due to the CPU limits:
    dataset["train"] = dataset["train"].select(range(10))
    dataset["test"] = dataset["test"].select(range(5))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_ds = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=LEARNING_RATE,         
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
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

    print("Training start...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./tmp_model")
    tokenizer.save_pretrained("./tmp_model")
    print("Model saved: './tmp_model'")

if __name__ == "__main__":
    train()