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
    conda deactivate
    ```

  * Ollama Singularity image: `/software/rhel9/manual/install/ollama/ollama-0.11.10.sif`

* **Database Access**: Obtain credentials for MySQL. Create a file called '.env' and save it to your CRCD home directory (/ihome/project_name/user_name/.env). The file should contain the following:

	```
	DB_USER=myuser
	DB_PASS=mypassword
	DB_HOST=db.example.edu
	DB_NAME=sspSB
    ```
- To restrict others from reading the file use `chmod 600 .env`	

* **Table Creation**: Create or confirm a results table (see Section 3.5).

### Per-Experiment Setup

* Choose an appropriate model (`llama3`, `llama3:70b`, etc.).
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
   # Example output (with $USER replaced with ajk77): 
   # CLUSTER: gpu
   # 		JOBID 	PARTITION     NAME     USER ST    TIME  NODES NODELIST(REASON)
   #		1230409      l40s ollama_0    ajk77  R    0:31      1 gpu-n55
   squeue -M gpu --me --name=ollama_0.11.10_server_job --states=R -h -O NodeList,Comment
   # Example output (hostname:port): 
   # gpu-n57 48362
   ```

3. Note the server URL: `http://<hostname>:<port>` (e.g., `http://gpu-n57:48362`).

4. (Required first time) Pull desired models:
   - Request an interactive SMP session: `srun -M smp -p smp -n4 --mem=16G -t0-01:00:00 --pty bash`
   - In the session:
     ```
     module load singularity/4.3.2
     singularity shell /software/rhel9/manual/install/ollama/ollama-0.11.10.sif
     export OLLAMA_HOST=<hostname>:<port>  # e.g., gpu-n57:48362
     ollama pull llama3
	 
	 # Additional helpful commands:
	 # (1) To list downloaded models:
     ollama list
	 # (2) To directly interact with a model:
	 ollama run llama3
	 # (3) To exit interaction
	 /bye
	 # (4) To exist singularity shell 
	 exit
     ```

**Note on Models**:
* This example uses llama3 as the model. A full list of models avaliable within CRCD's current singularity image is provided here:  https://github.com/ollama/ollama/tree/v0.11.10?tab=readme-ov-file#model-library
* Model install will fail if you exceed your 75GB allowance in ~/.ollama
* To remove a model use 'ollama rm model_name'

### 3.2. Connect to the Database and Fetch Notes
Use `pymysql` to connect and fetch notes in batches to handle large datasets efficiently.

```python
from dotenv import load_dotenv
import pymysql
import ollama
import os
from pydantic import BaseModel, ValidationError
import json
from typing import List

load_dotenv('/ihome/project_name/user_name/.env')  ## Update with path from Section 2.)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# Database connection function 
def db_connect():
    db_host = DB_HOST
    db_user = DB_USER
    db_password = DB_PASS
    db_name = DB_NAME
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
    response:  Literal["(1) Likely", "(2) Unlikely", "(3) Unable to say"]
    explanation: str  # Concise explanation
```

- **Best Practice**: This prevents malformed JSON from the LLM. Validate responses and handle errors (e.g., retry on failure).

### 3.4. Process Notes with LLM Using System Prompts
Connect to the Ollama server and process each note. Use a system prompt for consistent behavior.

```python
# Ollama client setup
OLLAMA_HOST = 'http://gpu-n57:48362'  # Replace with your hostname:port
client = ollama.Client(host=OLLAMA_HOST)

# Unique name for this experiment (stored in meta column of result database)
EXP_NAME = 'questionsPNAv1.1 on 2025-09-24'

# Model selection
MODEL = "llama3"

# System prompt for consistent behavior
SYSTEM_PROMPT = """
You are a medical analysis assistant. Analyze the note for pneumonia diagnosis.
Answer strictly as '(1) Likely', '(2) Unlikely', or '(3) Unable to say'.
Provide a concise explanation. Output as JSON: {"response": "...", "explanation": "..."}.
Do not add extra text.
"""

# Process a single note
def process_note(note_text: str, model: str, max_attempts: int = 2) -> dict:
    user_prompt = f"Does this note suggest pneumonia is a diagnosis? Note: {note_text}"
	
	def call_llm():
		response = client.chat(
			model=model,
			messages=[
				{'role': 'system', 'content': SYSTEM_PROMPT},
				{'role': 'user', 'content': user_prompt}
			]
		)
		return response['message']['content']
		
	attempts = 0 
	while attempts < max_attempts:  # retry when validation fails
		try:
			res_content = call_llm()
			res_dict = json.loads(res_content)  # Parse JSON
			validated = LLMResponse(**res_dict)  # Validate with Pydantic
			return validated.model_dump()
		except (json.JSONDecodeError, ValidationError) as e:
			attempts += 1
			print(f"Attempt {attempts} failed: {e}")
			if attempts < max_attempts:
				print("Retrying once...")
				
    # Fallback if both attempts fail
    return {
        "response": "(4) LLM output invalid",
        "explanation": "LLM output failed validation."
    }
```

- **Best Practices**:
  - **System Prompts**: Use 'system' role to set rules (e.g., output format), reducing hallucinations and ensuring structured responses.
  - **Pydantic Validation**: Automatically checks types and structure; handle errors gracefully (e.g., retry up to max_attempts times).
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
        """, (res['patientvisitid'], res['accession'], res['response'], res['explanation'], res['meta'])) 
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
    "model": MODEL,
    "system_prompt": SYSTEM_PROMPT.strip(),
    "batch_size": 100,
    "timestamp": datetime.now().isoformat(),
    "ollama_host": OLLAMA_HOST,
	"exp_name": EXP_NAME
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
		res['meta'] = EXPERIMENT_META["exp_name"]
        batch_results.append(res)

        # Log each processed note
        logging.info(f"Processed note {accession}")

    insert_results(batch_results)
    print(f"Processed batch of {len(batch)} notes.")
```

---

## 4. Cleanup

* Cancel Ollama job (`scancel -M gpu <JOBID>`)

---

## 5. Troubleshooting

* **Ollama not responding**: Check with `squeue` and restart job
* **GPU OOM**: Switch to smaller model or larger GPU node
* **DB errors**: Verify credentials, schema

---

# 6. Execution Options [WORK IN PROGRESS]

There are three common ways to run this workflow on Pitt CRC:

---

## **6.1. Interactive Jupyter Notebook (OnDemand)**

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
#SBATCH --cpus-per-task=8
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

