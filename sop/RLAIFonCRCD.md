# [DRAFT] SOP: Reinforcement Learning from AI Feedback on Center for Research Computing and Data (CRCD)

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to perform Reinforcement Learning from AI Feedback (RLAIF) for aligning Large Language Models (LLMs) on the University of Pittsburgh's CRCD infrastructure. RLAIF is similar to RLHF but replaces human annotations with AI-generated feedback (e.g., preferences or rewards from a stronger LLM). It typically involves: (1) generating or using AI-labeled preference data, (2) training a reward model from this data, and (3) using reinforcement learning (e.g., Proximal Policy Optimization or PPO) to fine-tune the policy model. This process leverages libraries like Hugging Face's TRL for RLAIF workflows on GPU nodes, fetches or generates preference data, trains the reward and policy models, and saves aligned checkpoints.

This SOP is based on Pitt CRCD's machine learning training documentation ([CRCD ML Guide](https://crc-pages.pitt.edu/user-manual/compute/ml/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Base response data (e.g., prompts with pairs of candidate responses) is available; AI feedback will be generated or pre-labeled (e.g., using Ollama or another LLM).
* Data is compliant with institutional policies.

**Notes:**

* RLAIF reduces the need for human labelers but requires a reliable AI feedback model to avoid bias amplification.
* For secure environments, consult [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/).
* Additional guidance for Generative AI is available at [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * PyTorch with CUDA support (via `module load pytorch/2.0.1`)
  * Ollama for AI feedback generation (see original LLM on CRCD SOP for setup).

* **Create a Virtual Environment**:
  Create a Conda environment with required packages (update `/path/to/your/env` to your project location):

  ```bash
  module load python/ondemand-jupyter-python3.11
  conda create --prefix=/path/to/your/env python=3.11
  source activate /path/to/your/env
  pip install transformers datasets torch accelerate trl peft ollama jupyterlab dotenv
  conda deactivate
  ```

* **Data Access**:
  Ensure access to your base dataset (e.g., prompts and response pairs). Create a `.env` file in your CRCD home directory (`/ihome/project_name/user_name/.env`) with paths or credentials:

  ```text
  DATA_PATH=/project/data/rlaif_base_responses.jsonl
  OLLAMA_HOST=http://gpu-n57:48362  ## From Ollama server setup
  ```

  * Restrict permissions: `chmod 600 .env`
  * View permissions: `ls -la`
  * View contents: `cat .env`

---

## 3. Implementation in Python

### 3.1. Start a GPU Training Job

1. Start Ollama server for AI feedback generation (if needed; see original SOP Section 3.1).

2. Submit a Slurm job for training. Choose based on GPU needs:

   * For L40S (48GB, single GPU):

     ```bash
     sbatch --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 your_rlaif_script.slurm
     ```

   * For A100 (80GB, multi-GPU example):

     ```bash
     sbatch --partition=a100 --gres=gpu:2 --mem=200G --cpus-per-task=32 your_rlaif_script.slurm
     ```

3. Query the job status:

   ```bash
   squeue -M gpu -u $USER
   ```

4. Note the node(s) for monitoring.

5. (Required first time) Pull base models via Hugging Face in an interactive session (similar to RLHF).

### 3.2. Customize the Python Script

This script includes: (1) generating AI preferences (if not pre-done), (2) training a reward model, and (3) PPO for policy alignment. Use Ollama for AI feedback.

```python
from dotenv import load_dotenv
import os
import ollama
from trl import RewardTrainer, PPOTrainer, PPOConfig
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')  ## Update with your path
DATA_PATH = os.getenv("DATA_PATH", "/project/data/rlaif_base_responses.jsonl")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://gpu-n57:48362")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Experiment Configuration
EXP_NAME = 'rlaif_v1 on 2025-10-01'  ## Replace with your experiment's name
BASE_MODEL = "gpt2"  ## Choose a base SFT model
FEEDBACK_MODEL = "llama3"  ## AI model for generating preferences
REWARD_MODEL_OUTPUT = f"./results/{EXP_NAME}/reward_model"
POLICY_MODEL_OUTPUT = f"./results/{EXP_NAME}/policy_model"

# Load tokenizer and base dataset (assumes {"prompt": str, "response_a": str, "response_b": str})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
base_dataset = load_dataset("json", data_files=DATA_PATH)

# Phase 0: Generate AI Preferences (if not already labeled)
def generate_ai_preference(prompt, response_a, response_b):
    feedback_prompt = f"Compare these two responses to the prompt '{prompt}':\nA: {response_a}\nB: {response_b}\nWhich is better? Respond with 'A' or 'B' and a brief reason."
    response = ollama_client.chat(
        model=FEEDBACK_MODEL,
        messages=[{'role': 'user', 'content': feedback_prompt}]
    )['message']['content']
    chosen = response_a if 'A' in response else response_b
    rejected = response_b if 'A' in response else response_a
    return chosen, rejected

def label_dataset(dataset):
    labeled_data = []
    for example in dataset["train"]:
        prompt = example["prompt"]
        resp_a = example["response_a"]
        resp_b = example["response_b"]
        chosen, rejected = generate_ai_preference(prompt, resp_a, resp_b)
        labeled_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(labeled_data)

# Preprocess for reward training
def preprocess_function(examples):
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

# Phase 1: Train Reward Model
def train_reward_model(processed_dataset):
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
    reward_trainer.train()
    reward_trainer.save_model(REWARD_MODEL_OUTPUT)
    return reward_trainer.model

# Phase 2: PPO for Policy Alignment
def train_policy(reward_model, base_dataset):
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
        ref_model=AutoModelForCausalLM.from_pretrained(BASE_MODEL),
        tokenizer=tokenizer,
        dataset=base_dataset["train"],  # Prompts for generation
    )

    # Simulate RL loop (generate, reward, update)
    for epoch in range(2):  # Adjust epochs
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=50)
            rewards = []
            for response in response_tensors:
                reward_input = tokenizer.decode(response)
                # Use trained reward model for scoring
                reward = reward_model(tokenizer.encode(reward_input, return_tensors="pt")).logits[0].item()
                rewards.append(reward)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            logging.info(stats)
    ppo_trainer.save_model(POLICY_MODEL_OUTPUT)

# Main function with logging
def main():
    logging.basicConfig(
        filename=f"rlaif_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info(f"Starting RLAIF: {EXP_NAME}")

    # Label with AI if needed (or load pre-labeled)
    labeled_dataset = label_dataset(base_dataset)  # Comment out if pre-labeled
    processed_dataset = labeled_dataset.map(preprocess_function, batched=True).train_test_split(test_size=0.1)

    # Train reward model
    reward_model = train_reward_model(processed_dataset)

    # Train policy with PPO
    train_policy(reward_model, base_dataset)

if __name__ == "__main__":
    main()
```

**Details to update include:**

* Path in `load_dotenv()`
* Experiment name in `EXP_NAME`
* Base and feedback models in `BASE_MODEL` and `FEEDBACK_MODEL`
* Dataset format and AI preference generation in `generate_ai_preference()`
* Reward computation in PPO loop
* Hyperparameters like epochs, batch sizes, and learning rates
* Add evaluation if available
* Logging and output directories
* If preferences are pre-generated, skip `label_dataset()` and load directly

---

## 4. Cleanup

* Cancel training job (JOBID from squeue):

  ```bash
  scancel -M gpu <JOBID>
  ```

* Cancel Ollama server if used (see original SOP Section 4).
* Remove large checkpoints and logs: `rm -rf ./results/* ./logs/*`

---

## 5. Troubleshooting

* **AI Feedback Quality**: If preferences are inconsistent, refine the `feedback_prompt` or use a stronger model.
* **GPU OOM**: Reduce batch sizes, use PEFT, or distribute across more GPUs.
* **Instability in PPO**: Tune KL penalty or learning rate; check for NaN in rewards.
* **Ollama Issues**: Ensure server is running; fallback to HF models for feedback if needed.
* **Dataset Errors**: Verify base data format; balance chosen/rejected pairs.

---

## 6. Execution Instructions

### 6.1. Jupyter Notebook (OnDemand)

Ideal for development, e.g., testing AI labeling and small-scale training.

**Steps:**

1. Start Ollama server (if needed) and GPU interactive session (e.g., `srun --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 -t0-04:00:00 --pty bash`).
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu) and launch Jupyter with your Conda env.
3. Edit and run the Python script in phases (e.g., labeling first).
4. Monitor with `nvidia-smi` or logs.
5. Cancel jobs post-completion.

### 6.2. Process Job in Terminal with `sbatch`

For full-scale RLAIF training.

**Steps:**

1. Start Ollama server if needed.
2. Create a Slurm script (e.g., `your_rlaif_script.slurm`) that loads modules, activates env, and runs the Python file.
3. Submit with `sbatch your_rlaif_script.slurm`.
4. Monitor with `squeue`, `sacct`, or tail logs.
5. Cancel post-completion.