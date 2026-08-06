import json

with open('train.json', 'w') as f:
    json.dump([{'text': 'Supermarket Market Research Report: Branch A achieved highest total revenue.'}, {'text': 'Customer Behavior Analysis: Loyalty cardholders spend 25% more per transaction.'}], f)

with open('val.json', 'w') as f:
    json.dump([{'text': 'Branch Performance Evaluation: Branch B shows strong grocery sales.'}], f)

with open('test_model.py', 'w') as f:
    f.write('''from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

prompt = 'Analyze the top-performing supermarket branch and customer behavior:'

print('--- BASE MODEL OUTPUT ---')
base_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
base_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
base_gen = pipeline('text-generation', model=base_model, tokenizer=base_tokenizer)
print(base_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text'])

print('\\n--- FINE-TUNED MODEL OUTPUT ---')
try:
    ft_tokenizer = AutoTokenizer.from_pretrained('./output')
    ft_model = AutoModelForCausalLM.from_pretrained('./output')
    ft_gen = pipeline('text-generation', model=ft_model, tokenizer=ft_tokenizer)
    print(ft_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text'])
except Exception as e:
    print('Fine-tuned checkpoint not found locally yet.')
''')
print('Files created successfully!')

