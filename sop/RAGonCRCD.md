# [DRAFT] SOP: Retrieval Augmented Generation on Center for Research Computing and Data (CRCD)

## 1. Purpose

This Standard Operating Procedure (SOP) provides a comprehensive guide to implementing Retrieval Augmented Generation (RAG) for Large Language Models (LLMs) on the University of Pittsburgh's CRCD infrastructure. RAG enhances LLM responses by retrieving relevant documents from a knowledge base (e.g., via vector search) and augmenting the prompt with this context, improving accuracy and reducing hallucinations. This process involves embedding documents into a vector store (e.g., using FAISS), querying for similar chunks during inference, and generating responses with an LLM (e.g., via Ollama or Hugging Face). It leverages GPU nodes for efficient embedding and inference, connects to data sources (e.g., files, databases), builds and queries the index, and outputs augmented generations.

This SOP is based on Pitt CRCD's GPU and ML documentation ([CRCD GPU Guide](https://crc-pages.pitt.edu/user-manual/compute/gpu/) and [CRCD ML Guide](https://crc-pages.pitt.edu/user-manual/compute/ml/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Knowledge base documents (e.g., text files, PDFs) are stored in an accessible file system or database (e.g., `/project/data/corpus/`).
* Data is compliant with institutional policies (e.g., de-identified if sensitive; HIPAA for clinical data).
* Basic familiarity with vector embeddings and LLMs.

**Notes:**

* RAG is modular: separate embedding (one-time or incremental) from querying/generation for efficiency.
* Use GPU for large-scale embedding; CPU for small corpora.
* For secure environments, consult [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/).
* Additional guidance for Generative AI: [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).
* This SOP uses LangChain for orchestration, FAISS for vector storage, and Ollama for LLM inference. Alternatives (e.g., Hugging Face Embeddings, Pinecone for distributed stores) can be adapted.

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * PyTorch with CUDA (via `module load pytorch/2.0.1`) for GPU embeddings.
  * Ollama Singularity image for LLM: `/software/rhel9/manual/install/ollama/ollama-0.11.10.sif`.

* **Create a Virtual Environment**:
  Create a Conda environment with required packages (update `/path/to/your/env` to your project location):

  ```bash
  module load python/ondemand-jupyter-python3.11
  conda create --prefix=/path/to/your/env python=3.11
  source activate /path/to/your/env
  pip install langchain langchain-community faiss-gpu sentence-transformers ollama pymysql pydantic jupyterlab dotenv  # faiss-cpu for non-GPU
  conda deactivate
  ```

* **Database or File Access**:
  If using a MySQL database for documents, obtain credentials. Create a `.env` file in your CRCD home directory (`/ihome/project_name/user_name/.env`):

  ```text
  DB_USER=myuser
  DB_PASS=mypassword
  DB_HOST=db.example.edu
  DB_NAME=my_database
  CORPUS_PATH=/project/data/corpus  # Alternative for file-based
  OLLAMA_HOST=http://gpu-n57:48362  # From Ollama server
  ```

  * Restrict permissions: `chmod 600 .env`
  * View permissions: `ls -la`
  * View contents: `cat .env`

* **Embeddings and Models**:
  - Pull embedding model (e.g., 'all-MiniLM-L6-v2') via Sentence Transformers in an interactive session.
  - Pull LLM model (e.g., llama3) via Ollama (see Section 3.1).

---

## 3. Implementation in Python

### 3.1. Start the Ollama Server on a GPU Node

Follow the original LLM on CRCD SOP:

1. Submit Slurm job (e.g., for L40S):

   ```bash
   sbatch /software/rhel9/manual/install/ollama/ollama-0.11.10_l40s.slurm
   ```

2. Get hostname and port:

   ```bash
   squeue -M gpu -u $USER
   squeue -M gpu --me --name=ollama_0.11.10_server_job --states=R -h -O NodeList,Comment
   ```

3. Update `OLLAMA_HOST` in script or `.env`.

4. Pull models in interactive session if needed:

   ```bash
   srun -M smp -p smp -n4 --mem=16G -t0-01:00:00 --pty bash
   module load singularity/4.3.2
   singularity shell /software/rhel9/manual/install/ollama/ollama-0.11.10.sif
   export OLLAMA_HOST=<hostname>:<port>
   ollama pull llama3
   ```

### 3.2. Build and Index the Knowledge Base (One-Time or Incremental)

For large corpora, run this as a separate GPU job to embed and save the vector store.

```python
from dotenv import load_dotenv
import os
import pymysql
from langchain.document_loaders import DirectoryLoader, PyPDFLoader  # Or CSVLoader, etc.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')  ## Update path
CORPUS_PATH = os.getenv("CORPUS_PATH", "/project/data/corpus")
DB_HOST = os.getenv("DB_HOST")
# ... (other DB vars if using database)

# Experiment Configuration
EXP_NAME = 'rag_index_v1 on 2025-10-01'  ## Update
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = f"./vector_stores/{EXP_NAME}"

# Database connection (if using DB; else skip)
def db_connect():
    return pymysql.connect(host=DB_HOST, user=os.getenv("DB_USER"), password=os.getenv("DB_PASS"), database=os.getenv("DB_NAME"))

def fetch_docs_from_db(table_name="documents"):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text FROM {table_name}")
    docs = [{"id": row[0], "text": row[1]} for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return docs

# Load and split documents
def load_and_split_docs(use_db=False):
    if use_db:
        raw_docs = fetch_docs_from_db()
        docs = [Document(page_content=doc["text"], metadata={"id": doc["id"]}) for doc in raw_docs]
    else:
        loader = DirectoryLoader(CORPUS_PATH, glob="**/*.txt", loader_cls=TextLoader)  # Or PyPDFLoader for PDFs
        raw_docs = loader.load()
        docs = raw_docs  # From langchain loaders
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    return split_docs

# Build vector store
def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})  # GPU
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

# Main indexing function
def main():
    logging.basicConfig(filename=f"rag_index_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level=logging.INFO)
    logging.info(f"Starting RAG indexing: {EXP_NAME}")
    
    docs = load_and_split_docs(use_db=False)  # Toggle for DB
    vector_store = build_vector_store(docs)
    logging.info(f"Indexed {len(docs)} chunks to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()
```

**Details to update:**

* Data loading: Adapt for your source (e.g., add PDF handling).
* Embedding model and chunk params.
* Use DB if needed.

### 3.3. Customize the RAG Query and Generation Script

Run this for inference, loading the pre-built index.

```python
from dotenv import load_dotenv
import os
import ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
VECTOR_STORE_PATH = "./vector_stores/rag_index_v1 on 2025-10-01"  ## From indexing

# Experiment Configuration
EXP_NAME = 'rag_query_v1 on 2025-10-01'
MODEL = "llama3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # Retrieved chunks

# Ollama client
client = ollama.Client(host=OLLAMA_HOST)

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})
vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

# Prompt template
template = """Answer based on context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM runnable (using Ollama)
def ollama_llm(input_prompt):
    response = client.chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': input_prompt}]
    )
    return response['message']['content']

# RAG chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
    | prompt
    | ollama_llm
    | StrOutputParser()
)

# Main query function
def main(query: str = "What is the main finding in document X?"):
    logging.basicConfig(filename=f"rag_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level=logging.INFO)
    logging.info(f"Starting RAG query: {EXP_NAME} for '{query}'")
    
    result = rag_chain.invoke(query)
    logging.info(f"Response: {result}")
    print(result)

if __name__ == "__main__":
    main()
```

**Details to update:**

* Query in `main()`.
* Prompt template for your domain.
* Integrate DB for dynamic corpora.
* Add reranking or hybrid search for advanced setups.

---

## 4. Cleanup

* Cancel Ollama job:

  ```bash
  scancel -M gpu <JOBID>
  ```

* Remove vector stores/logs if space-limited: `rm -rf ./vector_stores/* *.log`

---

## 5. Troubleshooting

* **Embedding OOM**: Use smaller models, batch processing, or CPU (faiss-cpu).
* **Retrieval Irrelevance**: Tune chunk size/overlap or use better embeddings (e.g., 'BAAI/bge-large-en').
* **Ollama Not Responding**: Check `squeue`; restart job.
* **FAISS Load Errors**: Ensure consistent embeddings; set `allow_dangerous_deserialization=True` cautiously.
* **Slow Queries**: Index on GPU; limit TOP_K.
* **Data Loading Issues**: Verify paths/credentials; handle large PDFs with memory mapping.

---

## 6. Execution Instructions

### 6.1. Jupyter Notebook (OnDemand)

For development, indexing small corpora, and testing queries.

**Steps:**

1. Start Ollama server (Section 3.1).
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu); launch Jupyter with your Conda env.
3. Run indexing script first, then query script.
4. Debug with logs or interactive queries.
5. Cancel Ollama post-completion.

### 6.2. Process Job in Terminal with `sbatch`

For large-scale indexing or batch queries.

**Steps:**

1. Start Ollama server.
2. Create Slurm scripts for indexing (`rag_index.slurm`) and querying (`rag_query.slurm`), loading modules/env.
3. Submit: `sbatch rag_index.slurm` then `sbatch rag_query.slurm`.
4. Monitor: `squeue`, tail logs.
5. Cancel Ollama post-completion.