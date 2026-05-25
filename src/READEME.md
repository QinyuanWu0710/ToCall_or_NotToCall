# To Call or Not to Call

Official code for **To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling**.

The paper studies when language models should use external tools, focusing on web search. Tool calls can help when a model lacks reliable internal knowledge, but they can also be redundant, noisy, or too expensive. This repository contains the evaluation framework, tool-calling harnesses, scoring code, and analysis scripts used to study tool-use decisions through the paper's three factors: **necessity**, **utility**, and **affordability**.

Paper: https://arxiv.org/abs/2605.00737

## Paper Overview

Agentic LLM systems often expose tools such as web search, but the central decision is still made by the model: should it call the tool for this query? The paper evaluates that decision from two complementary perspectives:

- **Normative perspective**: infer when tool calls are truly needed or useful from the best allocation of tool calls.
- **Descriptive perspective**: infer the model's self-perceived need and utility from its observed tool-use behavior.

The framework supports experiments where models answer questions with no search, optional search, forced search, cost-aware tool descriptions, and budget-aware tool descriptions. The repository also includes support for lightweight downstream analysis of perceived need, factuality, search behavior, and controller-style experiments.

## Repository Highlights

- Local vLLM evaluation for open-weight models.
- OpenAI API evaluation for GPT-style models.
- Web-search tool execution through FastMCP.
- Chat-template tool harness for local models that support function/tool-call formatting.
- OpenAI Responses API official tool harness.
- Entity, BFCL, and InVivo-style input data hooks.
- Incremental JSONL result writing and CSV scoring summaries.
- Analysis scripts for tool-use behavior, perceived need, and prediction experiments.

## Project Structure

```text
Tool_Call_Code/
├── data/                         # Entity, BFCL, and InVivo-style input files
├── results/                      # Experiment outputs
├── analyse/                      # Analysis and plotting scripts
├── entity_data_construction/     # Entity dataset construction utilities
├── src/
│   ├── main.py                   # CLI entry point
│   ├── tester.py                 # Generation, search, and scoring orchestration
│   ├── agent_loop.py             # Base local/vLLM agent loop
│   ├── chat_harness_agent_loop.py # Chat-template tool harness for local models
│   ├── openai_agent_loop.py      # OpenAI API chat-style loop
│   ├── openai_agent_official.py  # OpenAI Responses API official tool harness
│   ├── config.py                 # Prompts, tool descriptions, generation defaults
│   ├── tool_manager.py           # Tool registration and execution
│   ├── mcp_client.py             # FastMCP client
│   ├── mcp_server.py             # FastMCP web_search server
│   ├── scorer.py                 # Entity/BFCL scoring
│   ├── run_scoring.py            # Re-score saved JSONL outputs
│   ├── _common.sh                # Shared cluster-portable shell setup
│   ├── _script.sh                # Base Slurm experiment script
│   ├── _script_chat_harness.sh   # Chat-template harness script
│   └── _script_official_harness.sh # OpenAI official harness script
├── requirements.txt
└── README.md
```

## Setup

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

On a cluster, point the provided shell scripts at your Python interpreter:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
```

Set only the API keys needed for your run:

```bash
# Required for OpenAI models and entity scoring
export OPENAI_API_KEY=your_openai_key

# Required for Google/SerpAPI-backed web search
export SERPAPI_API_KEY=your_serpapi_key

# Required only when SEARCH_PROVIDER=brave
export BRAVE_API_KEY=your_brave_key
```

The shell scripts resolve paths from their own location and write by default to repo-local `logging/` and `results/` directories. Override paths as needed:

```bash
export DATA=/path/to/data.csv
export BASE_OUTPUT_DIR=/path/to/results
export LOG_DIR=/path/to/logging
export MODEL=/path/to/model-or-api-model-name
```

## Data Format

Input files can be CSV or JSON. Entity-style data should include `entity_text` or `entity`:

```csv
entity_text
Albert Einstein
Marie Curie
Nikola Tesla
```

For new tasks, update `ENTITY_QUERY_PROMPT_TEMPLATE` in `src/config.py` so each row is converted into the intended user query.

Bundled data entry points include:

- `data/Entity.csv`: entity knowledge queries.
- `data/BFCL.csv`: BFCL-style tool-use examples.
- `data/InvivoQuery.csv`: InVivo-style user queries.

## Running Experiments

The same framework supports three search conditions:

- **No search**: the model answers from parametric knowledge.
- **Auto search**: the model decides whether to call `web_search`.
- **Force search**: the model always calls `web_search`, but still writes its own search query.

### No Search

```bash
python src/main.py \
  --data data/Entity.csv \
  --model-name Qwen/Qwen3-4B \
  --tool-description main \
  --output-dir results
```

### Auto Search

```bash
python src/main.py \
  --data data/Entity.csv \
  --model-name Qwen/Qwen3-4B \
  --tool-description main \
  --web-search \
  --search-provider google \
  --output-dir results
```

### Force Search

```bash
python src/main.py \
  --data data/Entity.csv \
  --model-name Qwen/Qwen3-4B \
  --tool-description main \
  --web-search \
  --force-search \
  --search-provider google \
  --output-dir results
```

## Harness Modes

The evaluation pipeline can be run through multiple tool-calling formats.

| Mode | Best for | How to enable | Script |
|---|---|---|---|
| Base local loop | Local vLLM models with the original JSON decision prompt | default | `src/_script.sh` |
| Chat-template harness | Local vLLM models that should see tools in chat/function-call format | `AGENT_HARNESS=chat` | `src/_script_chat_harness.sh` |
| OpenAI official harness | OpenAI API models using the Responses API tool interface | `OPENAI_AGENT_BACKEND=official` | `src/_script_official_harness.sh` |

### Base Local/vLLM Script

```bash
export PYTHON_BIN=/path/to/python
export MODEL=/path/to/local/model-or-hf-id
export SERPAPI_API_KEY=your_serpapi_key
export OPENAI_API_KEY=your_openai_key_for_scoring

sbatch src/_script.sh
```

### Chat-Template Harness

`src/_script_chat_harness.sh` runs local or self-hosted vLLM models with the chat-template tool format implemented in `src/chat_harness_agent_loop.py`. Instead of flattening tool descriptions and results into plain prompts, this path uses tokenizer chat templates and a function schema for `web_search`.

When a model calls the tool, the final generation receives a chat history shaped like:

```text
system -> user -> assistant(tool_call) -> tool(result) -> assistant
```

```bash
export PYTHON_BIN=/path/to/python
export MODEL=/path/to/local/model-or-hf-id
export SERPAPI_API_KEY=your_serpapi_key
export OPENAI_API_KEY=your_openai_key_for_scoring

sbatch src/_script_chat_harness.sh
```

For a small direct smoke test:

```bash
PYTHON_BIN=python3 \
MODEL=Qwen/Qwen3-30B-A3B \
LIMIT=2 \
bash src/_script_chat_harness.sh
```

### OpenAI Official Tool Harness

`src/_script_official_harness.sh` runs OpenAI API models through `src/openai_agent_official.py`, which uses OpenAI's official Responses API tool path. This is useful when comparing the paper's framework against API-level tool-use behavior.

By default, the script sets:

```bash
OPENAI_AGENT_BACKEND=official
OPENAI_AGENT_TOOL_BACKEND=responses_url
MCP_SEARCH_PROVIDER=mcp-serp
```

For SerpAPI-backed MCP search, `SERPAPI_API_KEY` is enough to construct the MCP server URL. You can also provide `MCP_SERVER_URL` explicitly for another remote MCP server.

```bash
export PYTHON_BIN=/path/to/python
export MODEL=gpt-5.5
export OPENAI_API_KEY=your_openai_key
export SERPAPI_API_KEY=your_serpapi_key

bash src/_script_official_harness.sh
```

To use OpenAI's hosted web search instead of an MCP URL backend:

```bash
OPENAI_AGENT_TOOL_BACKEND=openai_web bash src/_script_official_harness.sh
```

## Tool-Description and Cost Experiments

The paper studies how tool-use decisions change when the model is told different things about tool cost and budget. These variants live in `AgentConfig.TOOL_DESCRIPTIONS` in `src/config.py`. The default key is `main`.

Common variants include:

```text
main
tool-cost-0
tool-cost-10
tool-cost-100
tool-cost-1000
tool-cost-10000
tool-cost-cheap
tool-cost-expensive
tool-cost-100-budget-aware
tool-cost-100-budget-aware-v2
main-perceived-need
main-perceived-need-v1
main-perceived-need-v2
```

Each run is saved under a matching subdirectory:

```text
results/
└── main/
    ├── *.jsonl
    └── *_summary.csv
```

To add a new variant, edit `TOOL_DESCRIPTIONS` in `src/config.py` and include the new key in the `TOOL_DESCRIPTIONS=(...)` array of the script you want to run.

## CLI Arguments

| Argument | Default | Description |
|---|---:|---|
| `--data` | required | Path to CSV/JSON input file |
| `--model-name` | required | Local model path, Hugging Face model id, or OpenAI model name |
| `--web-search` | off | Enable web search |
| `--force-search` | off | Always call `web_search` |
| `--search-provider` | `google` | `google`/SerpAPI or `brave` |
| `--search-api-key` | env var | Search API key override |
| `--tool-description` | `main` | Tool-description variant from `AgentConfig.TOOL_DESCRIPTIONS` |
| `--task` | `entity` | Scoring task: `entity` or `bfcl` |
| `--bfcl-gt-file` | None | Ground-truth JSONL for BFCL scoring |
| `--max-tokens` | auto | Max generated tokens |
| `--temperature` | config default | Sampling temperature |
| `--tensor-parallel-size` | `1` | vLLM tensor parallel size |
| `--output-dir` | `./results` | Root output directory |
| `--save-every` | `1` | Checkpoint frequency |
| `--limit` | None | Limit examples for testing |
| `--skip-scorer` | off | Run inference only |
| `--keep-in-memory` | off | Keep results in memory after writing |

## Output Format

Results are saved as JSONL, one object per example:

```json
{
  "entity": "Albert Einstein",
  "query": "In a paragraph, could you tell me what you know about Albert Einstein?",
  "response": "Albert Einstein was a theoretical physicist...",
  "model": "Qwen/Qwen3-4B",
  "tool_calls": [
    {
      "type": "tool_selection",
      "decision": {
        "needs_tool": true,
        "tool_name": "web_search",
        "tool_input": "Albert Einstein biography",
        "reasoning": "Need factual grounding."
      }
    },
    {
      "type": "tool_execution",
      "tool_name": "web_search",
      "tool_input": "Albert Einstein biography",
      "result": "..."
    }
  ],
  "iterations": 3,
  "tokens_generated": 312,
  "web_search_enabled": true
}
```

Scoring summaries are written as CSV files with columns such as `entity`, `query`, `model`, `search_called`, `score`, `correct_claims`, and `total_claims`.

## Analysis

Analysis scripts live in `analyse/`. They cover result comparison, perceived-need checks, predictor experiments, alignment analyses, factuality distributions, and plotting utilities. Most scripts expect existing JSONL/CSV outputs under `results/`; inspect the script-level constants or shell wrappers before launching long runs.

## Citation

If you use this repository, please cite:

```bibtex
@misc{wu2026callnotcallframework,
  title={To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling},
  author={Qinyuan Wu and Soumi Das and Mahsa Amani and Arijit Nag and Seungeon Lee and Krishna P. Gummadi and Abhilasha Ravichander and Muhammad Bilal Zafar},
  year={2026},
  eprint={2605.00737},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2605.00737}
}
```
