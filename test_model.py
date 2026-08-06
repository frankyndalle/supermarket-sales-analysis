from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

prompt = 'Analyze the top-performing supermarket branch and customer behavior:'

print('--- BASE MODEL OUTPUT ---')
base_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
base_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
base_gen = pipeline('text-generation', model=base_model, tokenizer=base_tokenizer)
base_output = base_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
print(base_output)

print('\n--- FINE-TUNED MODEL OUTPUT ---')
# Check for checkpoints inside output directory
checkpoint_dir = './output'
if os.path.exists(checkpoint_dir):
    subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if subdirs:
        latest_checkpoint = max(subdirs, key=os.path.getmtime)
        ft_path = latest_checkpoint
    else:
        ft_path = checkpoint_dir

    try:
        ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
        ft_model = AutoModelForCausalLM.from_pretrained(ft_path)
        ft_gen = pipeline('text-generation', model=ft_model, tokenizer=ft_tokenizer)
        ft_output = ft_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        print(ft_output)
    except Exception as e:
        print(f'Error loading fine-tuned model: {e}')
else:
    print('Fine-tuned checkpoint directory not found locally yet.')

