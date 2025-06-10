# Falcon-7B Medical Chatbot Fine-Tuning

This project fine-tunes the Falcon-7B-instruct model on medical dialogue data to create a doctor-patient consultation chatbot.

## Project Description

The model is based on Falcon-7B-instruct, fine-tuned using QLoRA (4-bit quantization with LoRA adapters) on a dataset of doctor-patient interactions. The goal is to create a model that can generate plausible medical responses to patient queries.

## Requirements

- Google Colab Pro (recommended for GPU access)
- Python 3.8+
- Required Python packages (install via `pip`):
torch
transformers
datasets
peft
bitsandbytes
matplotlib
# ## Dataset
The dataset consists of doctor-patient interactions with the following structure:
```{
  "input": "Patient query text",
  "output": "Doctor response text"
}
```
The data is automatically split into training and evaluation sets during loading.

Model Configuration
Base model: tiiuae/falcon-7B-instruct

Quantization: 4-bit NF4 with double quantization

## LoRA configuration:

Rank (r): 32

Alpha: 64

Target modules: Falcon-specific layers

Dropout: 0.05

Training Parameters
Batch size: 4 (with gradient accumulation steps of 2)

Learning rate: 2e-4

Max steps: 500

Evaluation steps: every 100 steps

Max sequence length: 512 tokens

## Usage
After training, you can use the model to generate responses:

python
from transformers import pipeline
```
medical_bot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
def generate_response(patient_query):
    prompt = f"### The following is a doctor's opinion on a patient's query:\n### Patient query: {patient_query}\n### Doctor opinion:"
    response = medical_bot(
        prompt,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return response[0]['generated_text']
```
## Results
The training process outputs:

Training loss curves

Evaluation metrics

Token length distribution statistics

Files
falcon_patient_chatbot.ipynb: Main training notebook

/falcon-patient-chatbot: Saved model checkpoints

/logs: Training logs

## Limitations
This is a prototype and not for real medical use

The model may generate incorrect or harmful medical information

Limited to the scope of the training data

## Future Improvements
Expand dataset with more diverse medical cases

Implement more rigorous evaluation metrics

Add safety checks for generated content

Deploy as an interactive web application

