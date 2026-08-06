from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

dataset = load_dataset('json', data_files={'train': 'train.json', 'val': 'val.json'})

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    result['labels'] = result['input_ids'].copy()
    return result

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained('distilgpt2')

training_args = TrainingArguments(
    output_dir='./output',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val']
)

trainer.train()
