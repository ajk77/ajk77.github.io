# SOP: Running Large Language Models on Center for Research Computing and Data

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to process a large volume of clinical or textual notes through a LLM on the University of Pittsburgh's CRCD infrastructure. The process leverages Ollama for LLM inference on GPU nodes, connects to a MySQL database to fetch notes, processes them with structured prompts, and inserts results back into the database. This ensures scalability, reproducibility, and compliance with best practices for handling sensitive data (e.g., de-identified notes).

Key features:

* **Scalability**: Handles large datasets by batching queries and using GPU resources.
* **Best Practices**: Incorporates Pydantic for output validation and system-level prompts for consistent LLM behavior.
* **Output Handling**: Results are inserted directly into the database instead of writing to files, reducing I/O overhead and improving data integrity.
* **Focus**: Python implementation, assuming access to Pitt CRC's GPU clusters (e.g., A100 or L40S partitions).

This SOP is based on guidance from the Pitt CRC Bioinformatics documentation ([https://crc-pages.pitt.edu/user-manual/bioinformatics/ollama/](https://crc-pages.pitt.edu/user-manual/bioinformatics/ollama/)).

**Assumptions**:

* You have a Pitt CRC account and access to GPU resources.
* Notes are stored in a MySQL database (e.g., `sspSB.PNA2_100_RadNotes` in the example).
* Data is de-identified and compliant with institutional policies (e.g., HIPAA).
* Ollama server is version 0.1.10 or later, installed as a Singularity image.

**High-Level Workflow:**

```
          ┌─────────────────┐
          │  MySQL Database │
          │ (clinical notes)│
          └────────┬────────┘
                   │ fetch batches
                   ▼
         ┌───────────────────┐
         │   Python Script   │
         │  (prompts + code) │
         └────────┬──────────┘
                  │ send requests
                  ▼
         ┌───────────────────┐
         │  Ollama LLM GPU   │
         │  (system prompt)  │
         └────────┬──────────┘
                  │ validated JSON
                  ▼
        ┌─────────────────────┐
        │  Pydantic Schema    │
        │ (validation + retry)│
        └────────┬────────────┘
                 │ insert results
                 ▼
         ┌────────────────────┐
         │   MySQL Database   │
         │ (results table)    │
         └────────────────────┘
```

---

## 2. Prerequisites

### One-Time Setup

* **Software and Modules**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)

  * Conda environment with required packages:

    ```bash
    module load python/ondemand-jupyter-python3.11
    conda create --prefix=/path/to/your/env python=3.11
    source activate /path/to/your/env
    pip install ollama pymysql pydantic
    source deactivate
    ```

  * Ollama Singularity image: `/software/rhel9/manual/install/ollama/ollama-0.11.10.sif`

* **Database Access**: Obtain credentials for MySQL.

* **Table Creation**: Create or confirm a results table (see Section 3.5).

### Per-Experiment Setup

* Choose an appropriate model (`llama3` vs. `llama3:70b`).
* Submit GPU job to launch Ollama server (see Section 3.1).
* Pull the required model if not already available.

---

## 3. Step-by-Step Implementation in Python

### 3.1. Start the Ollama Server on a GPU Node
1. Submit a Slurm job to run the Ollama server. Choose the appropriate template based on GPU needs:
   - For L40S (48GB): `sbatch /software/rhel9/manual/install/ollama/ollama-0.11.10_l40s.slurm`
   - For A100 (80GB): `sbatch /software/rhel9/manual/install/ollama/ollama-0.11.10_a100_80gb.slurm`
   
   This allocates 125GB RAM, 16 cores, and a GPU.

2. Query the job to get the hostname and port:
   ```
   squeue -M gpu -u $USER
   # Example output: JOBID 1230450, NODE gpu-n57
   squeue -M gpu --me --name=ollama_0.11.10_server_job --states=R -h -O NodeList,Comment
   # Example output: gpu-n57 48362 (hostname:port)
   ```

3. Note the server URL: `http://<hostname>:<port>` (e.g., `http://gpu-n57:48362`).

4. (Optional) Pull a model if not already available:
   - Request an interactive SMP session: `srun -M smp -p smp -n4 --mem=16G -t0-04:00:00 --pty bash`
   - In the session:
     ```
     module load singularity/4.3.2
     singularity shell /software/rhel9/manual/install/ollama/ollama-0.11.10.sif
     export OLLAMA_HOST=<hostname>:<port>  # e.g., gpu-n57:48362
     ollama pull llama3
     ```

### 3.2. Connect to the Database and Fetch Notes
Use `pymysql` to connect and fetch notes in batches to handle large datasets efficiently.

```python
import pymysql
import ollama
from pydantic import BaseModel, ValidationError
import json
from typing import List

# Database connection function (from notebook example)
def db_connect():
    db_host = "auview.ccm.pitt.edu"
    db_user = "your_username"  # Replace with your credentials
    db_password = "your_password"
    db_name = "sspSB"
    conn = pymysql.connect(host=db_host, user=db_user, password=db_password, 
                           database=db_name, port=3306, local_infile=1)
    return conn

# Fetch notes in batches
def fetch_notes(batch_size=100):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT patientvisitid, accession, note_text FROM sspSB.PNA2_100_RadNotes")
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        yield batch
    cursor.close()
    conn.close()
```

- **Best Practice**: Fetch in batches (e.g., 100 notes) to manage memory and allow checkpointing (e.g., commit after each batch).

### 3.3. Define Structured Output with Pydantic
Use Pydantic to enforce a schema for LLM responses, ensuring consistency and validation.

```python
# Pydantic model for structured output
class LLMResponse(BaseModel):
    response: str  # e.g., '(1) Likely', '(2) Unlikely', or '(3) Unable to say'
    explanation: str  # Concise explanation
```

- **Best Practice**: This prevents malformed JSON from the LLM. Validate responses and handle errors (e.g., retry on failure).

### 3.4. Process Notes with LLM Using System Prompts
Connect to the Ollama server and process each note. Use a system prompt for consistent behavior.

```python
# Ollama client setup
OLLAMA_HOST = 'http://gpu-n57:48362'  # Replace with your hostname:port
client = ollama.Client(host=OLLAMA_HOST)

# System prompt for consistent behavior
SYSTEM_PROMPT = """
You are a medical analysis assistant. Analyze the note for pneumonia diagnosis.
Answer strictly as '(1) Likely', '(2) Unlikely', or '(3) Unable to say because of lack of information'.
Provide a concise explanation. Output as JSON: {"response": "...", "explanation": "..."}.
Do not add extra text.
"""

# Process a single note
def process_note(note_text: str, model: str = 'llama3') -> dict:
    user_prompt = f"Does this note suggest pneumonia is a diagnosis? Note: {note_text}"
    
    response = client.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    
    res_content = response['message']['content']
    try:
        res_dict = json.loads(res_content)  # Parse JSON
        validated = LLMResponse(**res_dict)  # Validate with Pydantic
        return validated.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error processing note: {e}")
        return {"response": "(3) Unable to say", "explanation": "LLM output invalid; retry needed."}  # Fallback
```

- **Best Practices**:
  - **System Prompts**: Use 'system' role to set rules (e.g., output format), reducing hallucinations and ensuring structured responses.
  - **Pydantic Validation**: Automatically checks types and structure; handle errors gracefully (e.g., retry up to 3 times).
  - **Error Handling**: Wrap in try-except for robustness. Log errors to stdout or a table.
  - **Model Selection**: Start with smaller models like `llama3` for speed; scale to larger for accuracy.
  - **Batching**: Process in loops over fetched batches to avoid timeouts.

### 3.5. Insert Results Back into the Database
Create or use a results table (e.g., `sspSB.PNA_Results`) with columns: `patientvisitid`, `accession`, `llm_response`, `llm_explanation`, `query_meta`.

```python
# Assume table creation (run once)
def create_results_table():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sspSB.PNA_Results (
            patientvisitid INT,
            accession VARCHAR(255),
            llm_response VARCHAR(50),
            llm_explanation TEXT,
            query_meta VARCHAR(255)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Insert batch results
def insert_results(results: List[dict]):
    conn = db_connect()
    cursor = conn.cursor()
    for res in results:
        cursor.execute("""
            INSERT INTO sspSB.PNA_Results (patientvisitid, accession, llm_response, llm_explanation, query_meta)
            VALUES (%s, %s, %s, %s, %s)
        """, (res['patientvisitid'], res['accession'], res['response'], res['explanation'], 
              'questionsPNAv1.1 on 2025-09-24'))  # Update meta as needed
    conn.commit()
    cursor.close()
    conn.close()
```

- **Best Practice**: Use parameterized queries to prevent SQL injection. Commit after each batch for atomicity.

---

### 3.6. Main Processing Loop with Experiment Logging

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f"llm_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Record experiment metadata
EXPERIMENT_META = {
    "model": "llama3",
    "system_prompt": SYSTEM_PROMPT.strip(),
    "batch_size": 100,
    "timestamp": datetime.now().isoformat(),
    "ollama_host": OLLAMA_HOST
}
logging.info(f"Experiment setup: {EXPERIMENT_META}")

create_results_table()  # Run once

for batch in fetch_notes(batch_size=EXPERIMENT_META["batch_size"]):
    batch_results = []
    for record in batch:
        patientvisitid, accession, note_text = record
        note_text = note_text.decode('utf-8') if isinstance(note_text, bytes) else note_text
        res = process_note(note_text, model=EXPERIMENT_META["model"])
        res['patientvisitid'] = patientvisitid
        res['accession'] = accession
        batch_results.append(res)

        # Log each processed note
        logging.info(f"Processed note {accession} with response {res['response']}")

    insert_results(batch_results)
    print(f"Processed batch of {len(batch)} notes.")
```

---

## 4. Additional Best Practices

* Prompt engineering guidance
* Retry/fallback for malformed JSON
* Use environment variables for DB credentials
* Parallelization with job arrays for very large datasets

---

## 5. Cleanup

* Cancel Ollama job (`scancel -M gpu <JOBID>`)
* Close DB connections
* Remove unused models

---

## 6. Troubleshooting

* **Ollama not responding**: Check with `squeue` and restart job
* **GPU OOM**: Switch to smaller model or larger GPU node
* **DB errors**: Verify credentials, schema

---

## 7. Running as a Slurm Batch Job

To avoid staying in Jupyter, wrap the Python script into an `sbatch` file.

**Example: `run_llm_pipeline.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=llm_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=llm_pipeline_%j.out
#SBATCH --error=llm_pipeline_%j.err

# Load environment
module load python/ondemand-jupyter-python3.11
source activate /path/to/your/env

# Start Ollama server in background
module load singularity/4.3.2
singularity exec /software/rhel9/manual/install/ollama/ollama-0.11.10.sif \
    ollama serve --host 0.0.0.0:11434 &

# Wait a few seconds for server to start
sleep 10

# Run the Python pipeline
python run_llm_pipeline.py

# Cleanup
echo "Job completed at $(date)"
```

**Usage:**

```bash
sbatch run_llm_pipeline.slurm
```

* **Notes**:

  * Update `python run_llm_pipeline.py` to point to your actual script.
  * By default, Ollama listens on port `11434`. Update if you’re using a dynamic port.
  * Log files (`.out`, `.err`, `.log`) provide an audit trail.

---


# 7. Execution Options

There are three common ways to run this workflow on Pitt CRC:

---

## **7.1. Interactive Jupyter Notebook (OnDemand)**

This is the most user-friendly option for **testing, debugging, and prompt development**. You request an interactive GPU session, start an Ollama server, and run your code inside a Jupyter notebook.

**Steps:**

1. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu).
2. Request an interactive GPU session (e.g., A100 or L40S).
3. In the JupyterLab terminal, start Ollama:

```bash
module load singularity/4.3.2
singularity exec /software/rhel9/manual/install/ollama/ollama-0.11.10.sif ollama serve --host 0.0.0.0:11434 &
```

4. In a notebook cell, run your Python pipeline:

```python
%run run_llm_pipeline.py
```

**When to use:**

* Small pilot runs (e.g., 100–1,000 notes).
* Prompt engineering experiments.
* Debugging Pydantic schema or retry logic.
* Fast iteration with visualizations and ad hoc queries.

---

## **7.2. Batch Job with `sbatch` (Single Job)**

This runs the entire pipeline **unattended** in a single job. You create a `.slurm` script and submit it with `sbatch`.

**Example: `run_llm_pipeline.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=llm_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=llm_pipeline_%j.out
#SBATCH --error=llm_pipeline_%j.err

module load python/ondemand-jupyter-python3.11
source activate /path/to/your/env
module load singularity/4.3.2

# Start Ollama server in background
singularity exec /software/rhel9/manual/install/ollama/ollama-0.11.10.sif \
    ollama serve --host 0.0.0.0:11434 &
sleep 10

# Run the pipeline
python run_llm_pipeline.py
```

Submit with:

```bash
sbatch run_llm_pipeline.slurm
```

**When to use:**

* Medium-to-large runs (10,000–500,000 notes).
* You want reproducibility and logs.
* The job can finish within the walltime (≤ 24h typical limit).
* Best balance of ease and automation.

---

## **7.3. Batch Job with Slurm Job Array (Parallelized)**

A job array splits the workload into **independent parallel jobs**, each handling a subset of the dataset. This scales best for **millions of notes**.

**Example: `run_llm_pipeline_array.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=llm_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-9        # 10 jobs in the array
#SBATCH --output=llm_array_%A_%a.out
#SBATCH --error=llm_array_%A_%a.err

module load python/ondemand-jupyter-python3.11
source activate /path/to/your/env
module load singularity/4.3.2

# Start Ollama server in background
singularity exec /software/rhel9/manual/install/ollama/ollama-0.11.10.sif \
    ollama serve --host 0.0.0.0:11434 &
sleep 10

# Each job gets a different chunk of data
CHUNK_SIZE=10000
OFFSET=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

python run_llm_pipeline.py --offset $OFFSET --limit $CHUNK_SIZE
```

Submit with:

```bash
sbatch run_llm_pipeline_array.slurm
```

**When to use:**

* Very large runs (500,000–10,000,000+ notes).
* Need to distribute across many GPUs in parallel.
* Each array task independently processes a dataset chunk.
* Useful if a single job would exceed walltime or memory.

