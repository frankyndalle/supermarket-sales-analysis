import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load the held-out evaluation prompts created in Step 1
with open("evaluation_prompts.json", "r") as f:
    eval_data = json.load(f)

# 2. Setup tokenizers and models
base_model_name = "distilgpt2"
finetuned_path = "./output/checkpoint-final"

print("Loading base model and fine-tuned model for comparison...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)

results = []

# 3. Run side-by-side generation for each prompt
for item in eval_data:
    prompt_text = item["prompt"]
    expected = item["expected_response"]
    
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    
    # Base model output
    base_outputs = base_model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
    base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    
    # Fine-tuned model output
    ft_outputs = finetuned_model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
    ft_text = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
    
    comparison = {
        "prompt": prompt_text,
        "expected_response": expected,
        "base_model_output": base_text,
        "finetuned_model_output": ft_text
    }
    results.append(comparison)

# 4. Save the comparison outputs to a JSON file (Requirement #4)
with open("./output/model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Comparison complete! Results saved to ./output/model_comparison_results.json")