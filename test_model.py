from transformers import pipeline

print("--- Loading Your Fine-Tuned Model ---")
# Point directly to the checkpoint folder where the weights are stored
generator = pipeline("text-generation", model="./output/checkpoint-500")

prompt = "Transaction in Yangon Branch A by Member buying Health and beauty"
result = generator(prompt, max_length=50, num_return_sequences=1)

print("\n--- Generation Result ---")
print(result[0]['generated_text'])