# [DRAFT] SOP: Supervised Fine Tuning on Center for Research Computing and Data (CRCD)

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to perform Supervised Fine-Tuning (SFT) of a Large Language Model (LLM) on the University of Pittsburgh's CRCD infrastructure. The process uses Hugging Face Transformers for fine-tuning on labeled data, fetches data from a source, trains with a loss function, and saves the tuned model.

This SOP draws from Pitt CRCD's ML training examples ([CRCD ML Guide](https://crc-pages.pitt.edu/user-manual/compute/ml/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Labeled data is in a file or database (e.g., JSONL format).
* Data complies with policies.

**Notes:**

* SFT requires less data than pretraining but still GPU-heavy.
* See [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/) for secure data.
* GenAI guidance: [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * PyTorch (via `module load pytorch/2.0.1`)

* **Create a Virtual Environment**:

  Similar to pretraining, but add `peft` for efficient tuning: `pip install transformers datasets torch accelerate peft jupyterlab dotenv`

* **Data Access**:
  `.env` with `DATA_PATH=/project/data/sft_dataset.jsonl`

---

## 3. Implementation in Python

### 3.1. Start a GPU Training Job

Similar to pretraining; use Slurm for GPU allocation.

### 3.2. Customize the Python Script

```python
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')
DATA_PATH = os.getenv("DATA_PATH")

# Experiment Configuration
EXP_NAME = 'sft_v1 on 2025-09-24'
MODEL_NAME = "bert-base-uncased"  ## For classification example

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

dataset = load_dataset("json", data_files=DATA_PATH)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"./results/{EXP_NAME}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

def main():
    logging.basicConfig(filename=f"sft_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level=logging.INFO)
    logging.info(f"Starting SFT: {EXP_NAME}")
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
```

**Details to update include:**

* Similar to pretraining, adapt for your task (e.g., causal LM for generation).

---

## 4. Cleanup

Similar to pretraining.

---

## 5. Troubleshooting

* **Overfitting**: Monitor eval loss.
* **Label mismatches**: Check dataset format.

---

## 6. Execution Instructions

Similar to pretraining, with Jupyter for dev and sbatch for runs.

