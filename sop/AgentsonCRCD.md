# [DRAFT] SOP: Agentic Workflows on Center for Research Computing and Data (CRCD)

## 1. Purpose

This Standard Operating Procedure (SOP) outlines the steps to implement agentic workflows for Large Language Models (LLMs) on the University of Pittsburgh's CRCD infrastructure. Agentic workflows enable LLMs to act autonomously as agents, breaking down complex tasks into steps, using tools for external interactions (e.g., APIs, databases, or computations), and delegating to sub-agents for specialized subtasks. This process leverages libraries like LangGraph (for workflow graphs) and LangChain (for tool integration) on GPU or CPU nodes, defines tools and sub-agents, executes multi-step reasoning, and logs outcomes.

This expanded SOP provides a more complete example, including detailed Python code for defining tools (e.g., a math solver and web search) and sub-agents (e.g., a research sub-agent and a summarizer sub-agent). It demonstrates a workflow for a research query, where the main agent plans, delegates, and synthesizes results.

This SOP is based on Pitt CRCD's machine learning and workflow documentation ([CRCD ML Guide](https://crc-pages.pitt.edu/user-manual/compute/ml/)).

**Assumptions:**

* You have a Pitt CRCD account.
* Input data (e.g., user queries) is provided via files or databases.
* Tools and sub-agents comply with institutional policies (e.g., no unauthorized API calls).

**Notes:**

* Agentic workflows can be iterative and stateful; use checkpoints for long-running tasks.
* For GPU-accelerated inference, integrate with Ollama or Hugging Face.
* Consult [CRCD SRE](https://crc-pages.pitt.edu/user-manual/getting-started/access_sre/) for secure setups.
* GenAI guidance: [Pitt Digital GenAI](https://www.digital.pitt.edu/acceptable-use-generative-artificial-intelligence-tools).

---

## 2. Prerequisites

### One-Time Setup

* **Existing CRCD Resources**:

  * Python 3.11+ (via `module load python/ondemand-jupyter-python3.11`)
  * PyTorch or similar for optional ML tools (via `module load pytorch/2.0.1`)

* **Create a Virtual Environment**:
  Create a Conda environment with required packages (update `/path/to/your/env` to your project location):

  ```bash
  module load python/ondemand-jupyter-python3.11
  conda create --prefix=/path/to/your/env python=3.11
  source activate /path/to/your/env
  pip install langchain langgraph langchain-community langchain-openai ollama requests beautifulsoup4 jupyterlab dotenv  # Add API keys if needed (e.g., for real web search)
  conda deactivate
  ```

* **Data and API Access**:
  Create a `.env` file in your CRCD home directory (`/ihome/project_name/user_name/.env`) with any API keys or paths:

  ```text
  OLLAMA_HOST=http://gpu-n57:48362  ## For LLM inference
  # Optional: OPENAI_API_KEY=your_key_here  ## For external APIs if used
  ```

  * Restrict permissions: `chmod 600 .env`
  * View permissions: `ls -la`
  * View contents: `cat .env`

---

## 3. Implementation in Python

### 3.1. Start a Compute Job

1. For GPU-accelerated LLM inference, start an Ollama server (see original LLM on CRCD SOP Section 3.1).

2. Submit a Slurm job for the workflow. Use CPU for light workflows or GPU for heavy inference:

   * CPU example:

     ```bash
     sbatch --partition=smp --mem=64G --cpus-per-task=8 your_agent_script.slurm
     ```

   * GPU example (for LLM calls):

     ```bash
     sbatch --partition=l40s --gres=gpu:1 --mem=125G --cpus-per-task=16 your_agent_script.slurm
     ```

3. Query the job status:

   ```bash
   squeue -M gpu -u $USER  # Or squeue -M smp for CPU
   ```

4. Note the node for monitoring.

### 3.2. Customize the Python Script

This expanded example uses LangGraph to define a graph-based workflow. The main agent (planner) decomposes a research query (e.g., "Summarize recent advancements in quantum computing"), delegates to sub-agents (researcher for gathering info, summarizer for condensing), and uses tools (math solver for calculations, web scraper for info retrieval). The script includes state management for tracking progress.

```python
from dotenv import load_dotenv
import os
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSearchAPIWrapper  # Optional; mock for no API
from typing import TypedDict, Annotated, List
import operator
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging

# Load environmental variables
load_dotenv('/ihome/project_name/user_name/.env')  ## Update with your path
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://gpu-n57:48362")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Experiment Configuration
EXP_NAME = 'agent_workflow_v2 on 2025-10-01'  ## Replace with your experiment's name
MODEL = "llama3"  ## LLM for agents

# Define Tools
@tool
def math_solver(equation: str) -> str:
    """Solve simple math equations. Input: a string equation like '2 + 2 * 3'."""
    try:
        result = eval(equation)  # Use safely; in production, use sympy for security
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_scraper(url: str) -> str:
    """Scrape and extract text from a webpage. Input: URL string."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text[:2000]  # Truncate for brevity
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Mock Google Search if no API (replace with real if keys available)
@tool
def web_search(query: str) -> str:
    """Search the web for information. Input: query string."""
    # Mock response; in production, use GoogleSearchAPIWrapper()
    return "Mock search results: Recent quantum advancements include error-corrected qubits from Google (2023)."

tools = [math_solver, web_scraper, web_search]

# Define Sub-Agents (using ReAct pattern via LangGraph)
def create_sub_agent(system_prompt: str):
    """Factory to create a ReAct sub-agent with tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    llm_runnable = RunnableLambda(lambda x: ollama_client.chat(
        model=MODEL,
        messages=[{'role': 'system', 'content': system_prompt}] + x['messages']
    )['message']['content'])
    return create_react_agent(llm_runnable, tools, prompt=prompt)

research_sub_agent = create_sub_agent(
    "You are a research sub-agent. Gather information using tools like web_search or web_scraper."
)

summarizer_sub_agent = create_sub_agent(
    "You are a summarizer sub-agent. Condense input into key points."
)

# Define State for Workflow
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], operator.add]
    research_results: str
    summary: str

# Workflow Nodes
def planner_node(state: AgentState):
    prompt = "Decompose the query into steps: 1. Research, 2. Summarize."
    response = ollama_client.chat(
        model=MODEL,
        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': state['messages'][-1].content}]
    )['message']['content']
    return {"messages": [HumanMessage(content=response)]}

def research_node(state: AgentState):
    result = research_sub_agent.invoke({"messages": state['messages']})
    return {"research_results": result['messages'][-1].content}

def summarize_node(state: AgentState):
    input_msg = HumanMessage(content=f"Summarize: {state['research_results']}")
    result = summarizer_sub_agent.invoke({"messages": [input_msg]})
    return {"summary": result['messages'][-1].content}

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", research_node)
workflow.add_node("summarizer", summarize_node)

# Edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# Main function with logging
def main(query: str = "Summarize recent advancements in quantum computing"):
    logging.basicConfig(
        filename=f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info(f"Starting agentic workflow: {EXP_NAME} for query: {query}")

    inputs = {"messages": [HumanMessage(content=query)]}
    result = app.invoke(inputs)
    logging.info(f"Final summary: {result['summary']}")
    print(result['summary'])

if __name__ == "__main__":
    main()
```

**Details to update include:**

* Path in `load_dotenv()`
* Experiment name in `EXP_NAME`
* LLM model in `MODEL`
* Tool definitions (e.g., add real APIs like Google Search with keys)
* Sub-agent prompts and tools
* Workflow nodes and edges for your task (e.g., add decision nodes for routing)
* State TypedDict for additional tracking
* Input query in `main()`
* Logging and output handling

---

## 4. Cleanup

* Cancel job (JOBID from squeue):

  ```bash
  scancel -M gpu <JOBID>  # Or scancel -M smp
  ```

* Cancel Ollama server if used (see original SOP Section 4).
* Remove logs: `rm -rf *.log`

---

## 5. Troubleshooting

* **Tool Errors**: Ensure dependencies (e.g., requests) are installed; handle API rate limits.
* **Graph Cycles**: Add conditions to edges to prevent infinite loops.
* **LLM Instability**: Refine prompts; use retries for flaky responses.
* **Memory Issues**: Reduce state size or use CPU for non-inference parts.
* **Sub-Agent Isolation**: If sub-agents share state unexpectedly, use separate graphs.

---

## 6. Execution Instructions

### 6.1. Jupyter Notebook (OnDemand)

For development and testing workflows.

**Steps:**

1. Start Ollama server (if needed) and interactive session (e.g., `srun --partition=l40s --gres=gpu:1 ...`).
2. Log into [OnDemand at Pitt CRC](https://ondemand.htc.crc.pitt.edu); launch Jupyter with your Conda env.
3. Edit and run the script; visualize graph with `app.get_graph().draw_mermaid()`.
4. Cancel jobs post-completion.

### 6.2. Process Job in Terminal with `sbatch`

For production workflows.

**Steps:**

1. Start Ollama if needed.
2. Create Slurm script wrapping your Python file.
3. Submit with `sbatch`.
4. Monitor with `squeue` or logs.
5. Cancel post-completion.