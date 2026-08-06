# Supermarket Sales Analysis & Fine-Tuning (SBA 928)

This repository contains the complete reproducible code, training splits (train.json), validation splits (val.json), and evaluation scripts for the SBA 928 market-research fine-tuning assignment using DistilGPT-2.

## Project Structure
- train.json - Training dataset containing structured market-research prompts and expected responses.
- val.json - Validation dataset for evaluating generalization during training.
- train_model.py - Script to load the dataset, configure Hugging Face tokenization, and fine-tune DistilGPT-2.
- test_model.py - Evaluation script that performs a side-by-side comparison between the base DistilGPT-2 model and the fine-tuned market-research model.
- requirements.txt - Project dependencies.

## Installation & Setup

1. **Clone the repository:**
git clone https://github.com/frankyndalle/supermarket-sales-analysis.git
cd supermarket-sales-analysis

2. **Create and activate a virtual environment:**
python -m venv .venv
.venv\Scripts\Activate.ps1

3. **Install dependencies:**
pip install -r requirements.txt

## Execution Instructions

- **Run training:**
python train_model.py

- **Run the evaluation comparison:**
python test_model.py

## Training & Validation Results

The model was fine-tuned using train_model.py over 3 epochs with the captured loss metrics:
- Epoch 1 Validation Loss: 8.448
- Epoch 2 Validation Loss: 7.258
- Epoch 3 Validation Loss: 6.682 (Final Validation Loss)
- Overall Training Loss: 7.729

## Base vs. Fine-Tuned Model Evaluation Outputs

Running test_model.py executes side-by-side market-research prompts evaluating top-performing supermarket branches, payment methods, and customer behaviors across both models.
