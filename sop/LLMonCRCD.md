# SOP: Running Large Language Models on Center for Research Computing and Data

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to process clinical  notes through a LLM on the University of Pittsburgh's CRCD infrastructure. The process leverages Ollama for LLM inference on GPU nodes, connects to a MySQL database to fetch notes, processes them with structured prompts, and inserts results back into the database.

This SOP is based on guidance from the Pitt CRCD's Ollama documentation ([CRCD Ollama](https://crc-pages.pitt.edu/user-manual/bioinformatics/ollama/)).

**Assumptions**:
* You have a Pitt CRCD account.
* Notes are stored in a MySQL database (e.g., `sspSB.PNA2_100_RadNotes` in the example).
* Data is already de-identified and compliant with institutional policies (e.g., HIPAA). 

**Notes**:
* Ollama is not yet avaliable on CRCD's Secure Resarch Environment (SRE). For more information, please see ([CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/)).
* Additional guidance for use of Generative AI Tools outside of CRCD is provided by ([Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools)).

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:
  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * Current Ollama Singularity image: `/software/rhel9/manual/install/ollama/ollama-0.11.10.sif`

* **Create a Virtual Enviroment**: Create a Conda environment with required packages (update "/path/to/your/env" to your project location):
    ```bash
    module load python/ondemand-jupyter-python3.11
    conda create --prefix=/path/to/your/env python=3.11
    source activate /path/to/your/env
    pip install ollama pymysql pydantic
    conda deactivate
    ```

* **Database Access**: Obtain credentials for your project's MySQL database. Create a file called '.env' and save it to your CRCD home directory (/ihome/project_name/user_name/.env). The file should contain the following:
	```text
	DB_USER=myuser
	DB_PASS=mypassword
	DB_HOST=db.example.edu
	DB_NAME=sspSB
    ```
	* To restrict others from reading the file use `chmod 600 .env`	
	* To view file permissions use `ls -la`	
	* To view file contents use `cat .env`

---

## 3. Implementation in Python

### 3.1. First, Start the Ollama Server on a GPU Node
1. Submit a Slurm job to run the Ollama server. Choose the appropriate template based on GPU Memory needs:
   * For L40S (48GB): `sbatch /software/rhel9/manual/install/ollama/ollama-0.11.10_l40s.slurm`
   * For A100 (80GB): `sbatch /software/rhel9/manual/install/ollama/ollama-0.11.10_a100_80gb.slurm`
   
   This allocates 125GB RAM, 16 cores, and a GPU.

2. Query the job to get the hostname and port (replace $USER with your username):
   ```bash
   squeue -M gpu -u $USER
   # Example output (with $USER replaced with ajk77): 
   # CLUSTER: gpu
   # 		JOBID 	PARTITION     NAME     USER ST    TIME  NODES NODELIST(REASON)
   #		1230409      l40s ollama_0    ajk77  R    0:31      1 gpu-n55
   squeue -M gpu --me --name=ollama_0.11.10_server_job --states=R -h -O NodeList,Comment
   # Example output (hostname:port): 
   # gpu-n57 48362
   ```

3. Note the server URL: `http://<hostname>:<port>` (e.g., `http://gpu-n57:48362`). These numbers will need to be updated in your python script everytime you start a new ollama server. 

4. (Required the first time you use a model) Pull desired models (e.g., llama3):
   * Request an interactive SMP session: `srun -M smp -p smp -n4 --mem=16G -t0-01:00:00 --pty bash`
   * In the session:
    ```bash
    module load singularity/4.3.2
    singularity shell /software/rhel9/manual/install/ollama/ollama-0.11.10.sif
    export OLLAMA_HOST=<hostname>:<port>  # e.g., gpu-n57:48362
    ollama pull llama3
	``` 
	* Additional helpful commands within singularity shell:
		* To view downloaded models: `ollama list`
		* To directly interact with a model: `ollama run llama3`
		* To exit interaction: `/bye`
		* To exit singularity shell: `exit`
     
	* Note on Models:
		* This example uses llama3 as the model. A full list of models avaliable within CRCD's current singularity image is provided here: ([Ollama v0.11.10](https://github.com/ollama/ollama/tree/v0.11.10?tab=readme-ov-file#model-library)).
		* Model install will fail if you exceed your 75GB allowance in ~/.ollama
		* To remove a model (e.g., llama3), use `ollama rm llama3` within the sigularity shell.
		
### 3.2. Second, Customize the Python Script
	**Details to update include**:
	* the path passed into load_dotenv()
	* the hostname and port of your active Ollama node in OLLAMA_HOST
	* the experiment's name in EXP_NAME
	* the desired model in Model
	* the system prompt in SYSTEM_PROMPT
	* the Pydantic object defined in LLMResponse to match your system prompt
	* the database query in fetch_notes to pull from your source database
	* the user_prompt and fallback response in process_note())
	* the results table creation script in create_results_table()
	* the insert querey in insert_results ()

	```python
	from dotenv import load_dotenv
	import pymysql
	import ollama
	import os
	from pydantic import BaseModel, ValidationError
	import json
	import logging
	from datetime import datetime
	from typing import List

	# Load environmental variables
	load_dotenv('/ihome/project_name/user_name/.env')  ## Update with path from Section 2.)
	DB_USER = os.getenv("DB_USER")
	DB_PASS = os.getenv("DB_PASS")
	DB_HOST = os.getenv("DB_HOST")
	DB_NAME = os.getenv("DB_NAME")

	# Ollama client setup
	OLLAMA_HOST = 'http://gpu-n57:48362'  # Replace with your active hostname:port
	client = ollama.Client(host=OLLAMA_HOST)

	# Experiment Configureation
	EXP_NAME = 'questionsPNAv1.1 on 2025-09-24'
	MODEL = "llama3"

	# System prompt for consistent behavior
	SYSTEM_PROMPT = """
	You are a medical analysis assistant. Analyze the note for pneumonia diagnosis.
	Answer strictly as '(1) Likely', '(2) Unlikely', or '(3) Unable to say'.
	Provide a concise explanation. Output as JSON: {"response": "...", "explanation": "..."}.
	Do not add extra text.
	"""

	# Pydantic model for structured output (see Section 3.3.)
	class LLMResponse(BaseModel):
		response: Literal["(1) Likely", "(2) Unlikely", "(3) Unable to say"]
		explanation: str  # Concise explanation

	# Database connection function (see Section 3.2.)
	def db_connect():
		conn = pymysql.connect(
			host=DB_HOST,
			user=DB_USER,
			password=DB_PASS,
			database=DB_NAME,
			port=3306,
			local_infile=1)
		return conn

	# Fetch unprocessed notes in batches (see Section 3.2) 
	def fetch_notes(batch_size: int = 100, query_meta: str = EXP_NAME):
		conn = db_connect()
		cursor = conn.cursor()
		query = """
			SELECT r.patientvisitid, r.accession, r.note_text
			FROM sspSB.PNA2_100_RadNotes r
			LEFT JOIN sspSB.PNA_Results p
			ON r.patientvisitid = p.patientvisitid
			AND r.accession = p.accession
			AND p.query_meta = %s
			WHERE p.patientvisitid IS NULL
		"""	
		cursor.execute(query, (query_meta,))
		while True:
			batch = cursor.fetchmany(batch_size)
			if not batch:
				break
			yield batch
		cursor.close()
		conn.close()

	# Process a single note (see Section 3.4.)
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

	# Create reslts table 
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

	# Insert batch results (see Section 3.5)
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

	# Main processing loop with logging
	def main():
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

		create_results_table()  # Creates result table if needed

		for batch in fetch_notes(batch_size=EXPERIMENT_META["batch_size"], query_meta=EXPERIMENT_META["exp_name"]):
			batch_results = []
			for record in batch:
				patientvisitid, accession, note_text = record
				note_text = note_text.decode('utf-8') if isinstance(note_text, bytes) else note_text
				res = process_note(note_text, model=EXPERIMENT_META["model"])
				res['patientvisitid'] = patientvisitid
				res['accession'] = accession
				res['meta'] = EXPERIMENT_META["exp_name"]
				batch_results.append(res)
				logging.info(f"Processed note {accession}")
			insert_results(batch_results)
			print(f"Processed batch of {len(batch)} notes.")

	if __name__ == "__main__":
		main()

	```

---

## 4. Cleanup

* Cancel Ollama job: run `scancel -M gpu <JOBID>` in terminal. 

---

## 5. Troubleshooting

* **Ollama not responding**: Check with `squeue` and restart job
* **GPU OOM**: Switch to smaller model or larger GPU node
* **DB errors**: Verify credentials, schema

---

# 6. Execution Instructions

## **6.1. Jupyter Notebook (OnDemand)**

This is the most user-friendly option for **testing, debugging, and prompt development**. 

**Steps:**

1. Start Ollama server (see Section 3.1)
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu)
3. Edit the python script to meet your expeimental needs (see Section 3.2)
4. Run the script
5. At job completion, cancel the ollama server (see Section 4.)


## **6.2. Process job in terminal with sbatch**


**Steps:**

1. Start Ollama server (see Section 3.1)
2. Edit the python script to meet your expeimental needs (see Section 3.2)
3. Create a bash script for the job [An example will be added here in the future]
4. Run the bash script
5. At job completion, cancel the ollama server (see Section 4.)