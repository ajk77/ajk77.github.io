# [DRAFT] SOP: Continued Pretraining on Center for Research Computing and Data (CRCD) 

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to perform continued pretraining of a Large Language Model (LLM) on the University of Pittsburgh's CRCD infrastructure. The process uses Hugging Face Transformers for model training on GPU nodes, connects to a data source (e.g., a file system or database) to fetch pretraining data, processes it with tokenization and training loops, and saves checkpoints.

This SOP is inspired by Pitt CRCD's general GPU training documentation ([CRCD GPU Guide](https://crc-pages.pitt.edu/user-manual/compute/gpu/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Pretraining data is stored in a accessible file system or database (e.g., text files in `/project/data/`).
* Data is compliant with institutional policies (e.g., de-identified if sensitive).

**Notes:**

* Training is resource-intensive; monitor GPU usage to avoid out-of-memory errors.
* For secure environments, consult [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/).
* Additional guidance for AI training is available at [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * PyTorch with CUDA support (via `module load pytorch/2.0.1`)

* **Create a Virtual Environment**:
  Create a Conda environment with required packages (update `/path/to/your/env` to your project location):

  ```bash
  module load python/ondemand-jupyter-python3.11
  conda create --prefix=/path/to/your/env python=3.11
  source activate /path/to/your/env
  pip install transformers datasets torch accelerate jupyterlab dotenv
  conda deactivate
  ```

* **Data Access**:
  Ensure access to your pretraining dataset (e.g., a directory of text files). If using a database, create a `.env` file in your CRCD home directory (`/ihome/project_name/user_name/.env`) with credentials:

  ```text
  DATA_PATH=/project/data/pretrain_corpus
  ```

  * Restrict permissions: `chmod 600 .env`
  * View permissions: `ls -la`
  * View contents: `cat .env`

---

## 3. Implementation in Python

### 3.1. Start a GPU Training Job

1. Submit a Slurm job for training. Choose based on GPU needs:

   * For L40S (48GB):

     ```bash
     sbatch --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 your_training_script.slurm
     ```

   * For A100 (80GB):

     ```bash
     sbatch --partition=a100 --gres=gpu:1 --mem=200G --cpus-per-task=16 your_training_script.slurm
     ```

2. Query the job status:

   ```bash
   squeue -M gpu -u $USER
   ```

3. Note the node for monitoring.

4. (Required first time) Install or pull models via Hugging Face in an interactive session:

   ```bash
   srun -M smp -p smp -n4 --mem=16G -t0-01:00:00 --pty bash
   module load python/ondemand-jupyter-python3.11
   source activate /path/to/your/env
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
   ```

### 3.2. Customize the Python Script

```python
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')  ## Update with your path
DATA_PATH = os.getenv("DATA_PATH", "/project/data/pretrain_corpus")

# Experiment Configuration
EXP_NAME = 'pretrain_v1 on 2025-09-24'  ## Replace with your experiment's name
MODEL_NAME = "gpt2"  ## Choose a base model

# Training setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("text", data_files={"train": f"{DATA_PATH}/*.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./results/{EXP_NAME}",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Main training loop with logging
def main():
    logging.basicConfig(
        filename=f"pretrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info(f"Starting pretraining: {EXP_NAME}")
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
```

**Details to update include:**

* Path in `load_dotenv()`
* Experiment name in `EXP_NAME`
* Base model in `MODEL_NAME`
* Dataset path in `DATA_PATH`
* Tokenization and training args as needed
* Logging and output directories

---

## 4. Cleanup

* Cancel training job (JOBID from squeue):

  ```bash
  scancel -M gpu <JOBID>
  ```

* Remove large checkpoints if space is limited: `rm -rf ./results/*`

---

## 5. Troubleshooting

* **GPU OOM**: Reduce batch size or use gradient accumulation.
* **Dataset loading errors**: Verify file paths and formats.
* **Module conflicts**: Reload Python module if needed.

---

## 6. Execution Instructions

### 6.1. Jupyter Notebook (OnDemand)

Ideal for testing and small-scale pretraining.

**Steps:**

1. Start GPU interactive session (e.g., `srun --partition=l40s --gres=gpu:1 ...`).
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu) and launch Jupyter with your Conda env.
3. Edit and run the Python script.
4. Cancel GPU job post-completion.

### 6.2. Process Job in Terminal with `sbatch`

For production runs.

**Steps:**

1. Create a Slurm script wrapping your Python file.
2. Submit with `sbatch`.
3. Monitor with `squeue`.
4. Cancel post-completion.

