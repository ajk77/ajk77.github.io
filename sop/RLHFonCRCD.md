# [DRAFT] SOP: Reinforcement Learning from Human Feedback on Center for Research Computing and Data (CRCD)

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to perform Reinforcement Learning from Human Feedback (RLHF) for aligning Large Language Models (LLMs) on the University of Pittsburgh's CRCD infrastructure. RLHF typically involves two main phases: (1) training a reward model from human preference data, and (2) using reinforcement learning (e.g., Proximal Policy Optimization or PPO) to fine-tune the policy model based on rewards. This process leverages libraries like Hugging Face's TRL for RLHF workflows on GPU nodes, fetches preference data from a source, trains the reward and policy models, and saves aligned checkpoints.

This SOP is based on Pitt CRCD's machine learning training documentation ([CRCD ML Guide](https://crc-pages.pitt.edu/user-manual/compute/ml/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Preference data (e.g., pairs of prompts with chosen/rejected responses) is stored in an accessible file system or database (e.g., JSONL format in `/project/data/`).
* Data is compliant with institutional policies (e.g., de-identified if sensitive).

**Notes:**

* RLHF is computationally intensive and iterative; it requires careful monitoring to avoid instability.
* For secure environments, consult [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/).
* Additional guidance for Generative AI is available at [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).

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
  pip install transformers datasets torch accelerate trl peft jupyterlab dotenv
  conda deactivate
  ```

* **Data Access**:
  Ensure access to your preference dataset. Create a `.env` file in your CRCD home directory (`/ihome/project_name/user_name/.env`) with paths or credentials:

  ```text
  DATA_PATH=/project/data/rlhf_preferences.jsonl
  ```

  * Restrict permissions: `chmod 600 .env`
  * View permissions: `ls -la`
  * View contents: `cat .env`

---

## 3. Implementation in Python

### 3.1. Start a GPU Training Job

1. Submit a Slurm job for training. Choose based on GPU needs (RLHF often requires multi-GPU for efficiency):

   * For L40S (48GB, single GPU):

     ```bash
     sbatch --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 your_rlhf_script.slurm
     ```

   * For A100 (80GB, multi-GPU example):

     ```bash
     sbatch --partition=a100 --gres=gpu:2 --mem=200G --cpus-per-task=32 your_rlhf_script.slurm
     ```

2. Query the job status:

   ```bash
   squeue -M gpu -u $USER
   ```

3. Note the node(s) for monitoring.

4. (Required first time) Pull base models via Hugging Face in an interactive session:

   ```bash
   srun -M smp -p smp -n4 --mem=16G -t0-01:00:00 --pty bash
   module load python/ondemand-jupyter-python3.11
   source activate /path/to/your/env
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
   ```

### 3.2. Customize the Python Script

This script outlines a basic RLHF flow: first train a reward model (e.g., classifier on preferences), then use PPO for policy alignment. Adapt for your needs.

```python
from dotenv import load_dotenv
import os
from trl import RewardTrainer, PPOTrainer, PPOConfig
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')  ## Update with your path
DATA_PATH = os.getenv("DATA_PATH", "/project/data/rlhf_preferences.jsonl")

# Experiment Configuration
EXP_NAME = 'rlhf_v1 on 2025-10-01'  ## Replace with your experiment's name
BASE_MODEL = "gpt2"  ## Choose a base SFT model
REWARD_MODEL_OUTPUT = f"./results/{EXP_NAME}/reward_model"
POLICY_MODEL_OUTPUT = f"./results/{EXP_NAME}/policy_model"

# Load tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # For padding if needed
dataset = load_dataset("json", data_files=DATA_PATH)  # Assumes format: {"prompt": str, "chosen": str, "rejected": str}

def preprocess_function(examples):
    # Tokenize pairs for reward training
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer("Question: " + prompt + "\n\nAnswer: " + chosen, truncation=True)
        tokenized_rejected = tokenizer("Question: " + prompt + "\n\nAnswer: " + rejected, truncation=True)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
    return new_examples

processed_dataset = dataset.map(preprocess_function, batched=True)

# Phase 1: Train Reward Model
reward_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
reward_training_args = TrainingArguments(
    output_dir=REWARD_MODEL_OUTPUT,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_training_args,
    tokenizer=tokenizer,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
)

# Phase 2: PPO for Policy Alignment
def train_policy(reward_model):
    policy_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    # Optional: Use PEFT for efficiency
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    policy_model = get_peft_model(policy_model, peft_config)

    ppo_config = PPOConfig(
        model_name=BASE_MODEL,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=AutoModelForCausalLM.from_pretrained(BASE_MODEL),  # Reference for KL divergence
        tokenizer=tokenizer,
        dataset=dataset["train"],  # Prompts for generation
    )

    # Simulate RL loop (generate, reward, update)
    for epoch in range(2):  # Adjust epochs
        for batch in ppo_trainer.dataloader:
            # Generate responses
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=50)
            # Get rewards (using trained reward model)
            rewards = []  # Implement: pipe responses through reward_model
            for response in response_tensors:
                reward_input = tokenizer.decode(response)
                # Mock: reward = reward_model(reward_input).logits[0].item()
                rewards.append(0.0)  # Replace with actual
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            logging.info(stats)
    ppo_trainer.save_model(POLICY_MODEL_OUTPUT)

# Main function with logging
def main():
    logging.basicConfig(
        filename=f"rlhf_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info(f"Starting RLHF: {EXP_NAME}")

    # Train reward model
    reward_trainer.train()
    reward_trainer.save_model(REWARD_MODEL_OUTPUT)

    # Train policy with PPO
    train_policy(reward_model=reward_trainer.model)

if __name__ == "__main__":
    main()
```

**Details to update include:**

* Path in `load_dotenv()`
* Experiment name in `EXP_NAME`
* Base model in `BASE_MODEL`
* Dataset format and preprocessing in `preprocess_function()`
* Reward computation in PPO loop (integrate actual reward model inference)
* Hyperparameters like epochs, batch sizes, and learning rates
* Add evaluation datasets if available
* Logging and output directories

---

## 4. Cleanup

* Cancel training job (JOBID from squeue):

  ```bash
  scancel -M gpu <JOBID>
  ```

* Remove large checkpoints and logs if space is limited: `rm -rf ./results/* ./logs/*`

---

## 5. Troubleshooting

* **Instability in PPO**: Adjust KL penalty (via PPOConfig's kl_penalty or beta) or learning rate; monitor for NaN losses.
* **GPU OOM**: Reduce batch/mini-batch sizes, use PEFT (e.g., LoRA) for parameter-efficient tuning, or request more GPUs.
* **Dataset errors**: Verify preference data format; ensure chosen/rejected pairs are balanced.
* **Reward model accuracy**: If rewards are poor, increase reward training epochs or data quality.
* **Module conflicts**: Reload Python/PyTorch modules if import issues arise.

---

## 6. Execution Instructions

### 6.1. Jupyter Notebook (OnDemand)

Ideal for iterative development, testing reward modeling, and small-scale PPO runs.

**Steps:**

1. Start GPU interactive session (e.g., `srun --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 -t0-04:00:00 --pty bash`).
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu) and launch Jupyter with your Conda env.
3. Edit and run the Python script in sections (e.g., reward phase first).
4. Monitor with `nvidia-smi` or logs.
5. Cancel GPU job post-completion.

### 6.2. Process Job in Terminal with `sbatch`

For full-scale, long-running RLHF training.

**Steps:**

1. Create a Slurm script (e.g., `your_rlhf_script.slurm`) that loads modules, activates env, and runs the Python file.
2. Submit with `sbatch your_rlhf_script.slurm`.
3. Monitor with `squeue`, `sacct`, or tail logs.
4. Cancel post-completion if needed.