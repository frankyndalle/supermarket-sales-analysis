import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# 1. Load train and validation json files created in Step 1
with open("train.json", "r") as f:
    train_data = json.load(f)

with open("val.json", "r") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 2. Tokenizer setup using DistilGPT-2
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Model setup
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Training Arguments with Loss Tracking & Evaluation Strategy
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./output/logs",
    logging_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 5. Run Training & Capture Loss Results
print("Starting training process...")
train_result = trainer.train()
eval_results = trainer.evaluate()

# Save training and validation loss metrics to a log file (Requirement #5)
loss_metrics = {
    "train_loss": train_result.training_loss,
    "eval_loss": eval_results.get("eval_loss", 0.0)
}
with open("./output/training_loss_results.json", "w") as f:
    json.dump(loss_metrics, f, indent=4)

# 6. Save complete model weights and tokenizer (Requirement #2)
trainer.save_model("./output/checkpoint-final")
tokenizer.save_pretrained("./output/checkpoint-final")
print("Training complete! Full model weights, tokenizer, and loss results saved to ./output/")