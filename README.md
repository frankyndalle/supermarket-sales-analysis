# Supermarket Sales Analysis & Fine-Tuning (SBA 928)

This repository contains the complete reproducible code, training splits (`train.json`), validation splits (`val.json`), and evaluation scripts for the SBA 928 market-research fine-tuning assignment using DistilGPT-2.

## Project Structure
- `train.json` - Training dataset containing structured market-research prompts and expected responses.
- `val.json` - Validation dataset for evaluating generalization during training.
- `train_model.py` - Script to load the dataset, configure Hugging Face tokenization, and fine-tune DistilGPT-2.
- `test_model.py` - Evaluation script that performs a side-by-side comparison between the base DistilGPT-2 model and the fine-tuned market-research model.
- `requirements.txt` - Project dependencies.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/frankyndalle/supermarket-sales-analysis.git](https://github.com/frankyndalle/supermarket-sales-analysis.git)
   cd supermarket-sales-analysis
