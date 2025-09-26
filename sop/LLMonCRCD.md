# Standard Operating Procedure: Processing Large Numbers of Notes Through a Large Language Model (LLM) on Pitt CRC Using Ollama

## 1. Purpose
This Standard Operating Procedure (SOP) outlines the steps to process a large volume of clinical or textual notes through a Large Language Model (LLM) on the University of Pittsburgh's Center for Research Computing (Pitt CRC) infrastructure. The process leverages Ollama for LLM inference on GPU nodes, connects to a MySQL database to fetch notes, processes them with structured prompts, and inserts results back into the database. This ensures scalability, reproducibility, and compliance with best practices for handling sensitive data (e.g., de-identified notes).

Key features:
- **Scalability**: Handles large datasets by batching queries and using GPU resources.
- **Best Practices**: Incorporates Pydantic for output validation and system-level prompts for consistent LLM behavior.
- **Output Handling**: Results are inserted directly into the database instead of writing to files, reducing I/O overhead and improving data integrity.
- **Focus**: Python implementation, assuming access to Pitt CRC's GPU clusters (e.g., A100 or L40S partitions).

This SOP is based on guidance from the Pitt CRC Bioinformatics documentation (https://crc-pages.pitt.edu/user-manual/bioinformatics/ollama/) and the provided Jupyter notebook example (`Ollama_and_DB_connect.ipynb`).

**Assumptions**:
- You have a Pitt CRC account and access to GPU resources.
- Notes are stored in a MySQL database (e.g., `sspSB.PNA2_100_RadNotes` as in the example).
- Data is de-identified and compliant with institutional policies (e.g., HIPAA).
- Ollama server is version 0.1.10 or later, installed as a Singularity image.

## 2. Prerequisites
- **Software and Modules**:
  - Python 3.11+ (load via `module load python/ondemand-jupyter-python3.11` on Pitt CRC).
  - Install required Python packages in a Conda environment (e.g., `/ix1/bioinformatics/python_envs/ollama` as in the guidance):
    ```
    module load python/ondemand-jupyter-python3.11
    conda create --prefix=/path/to/your/env python=3.11
    source activate /path/to/your/env
    pip install ollama pymysql pydantic
    source deactivate
    ```
  - Ollama Singularity image: `/software/rhel9/manual/install/ollama/ollama-0.11.10.sif`.
- **Database Access**: Credentials for MySQL (e.g., host `auview.ccm.pitt.edu`, user, password, database `sspSB`).
- **Model Selection**: Choose an LLM model compatible with available GPU memory (e.g., `llama3` for 8B parameters on L40S 48GB nodes; larger models like `llama3:70b` may require A100 80GB).
- **Cluster Access**: Submit jobs via Slurm (e.g., `sbatch`).
- **Best Practice Tools**:
  - **Pydantic**: For defining structured output schemas and validating LLM responses.
  - **System Prompts**: To guide LLM behavior consistently across queries.

**Resource Considerations**:
- For large datasets (e.g., 1000+ notes), use batching to avoid overwhelming the LLM server.
- Monitor GPU usage with `squeue -M gpu -u $USER`.
- Estimated runtime: ~1-5 seconds per note, depending on model and note length.

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

### 3.6. Main Processing Loop
Tie it together in a script or Jupyter notebook.

```python
create_results_table()  # Run once

for batch in fetch_notes(batch_size=100):
    batch_results = []
    for record in batch:
        patientvisitid, accession, note_text = record
        note_text = note_text.decode('utf-8') if isinstance(note_text, bytes) else note_text
        res = process_note(note_text)
        res['patientvisitid'] = patientvisitid
        res['accession'] = accession
        batch_results.append(res)
    insert_results(batch_results)
    print(f"Processed batch of {len(batch)} notes.")
```

- Run this in a Jupyter session on OnDemand (ondemand.htc.crc.pitt.edu) or as a batch job on HTC/SMP for long-running tasks.

## 4. Additional Best Practices
- **Prompt Engineering**: Keep user prompts concise; use system prompts for invariants (e.g., format).
- **Rate Limiting**: Add `time.sleep(1)` between calls if the server overloads.
- **Monitoring**: Use `sstat` or `sacct` to track job metrics.
- **Error Retry**: Implement exponential backoff for Ollama failures.
- **Security**: Never hardcode credentials; use environment variables or secrets.
- **Testing**: Start with a small batch (e.g., 10 notes) to validate.
- **Scalability**: For millions of notes, parallelize with multiprocessing (e.g., one process per core), but monitor GPU sharing.

## 5. Cleanup
- Cancel the Ollama job: `scancel -M gpu <JOBID>`
- Close database connections in code.
- Remove temporary models if needed: `ollama rm <model>` via interactive session.

## 6. Troubleshooting
- Ollama not responding: Check `squeue` and restart job.
- GPU OOM: Switch to smaller model or larger GPU.
- DB Errors: Verify credentials and table schema.
- Contact Pitt CRC support for cluster issues.

This SOP ensures efficient, reliable processing. Update as Ollama/CRC versions change.