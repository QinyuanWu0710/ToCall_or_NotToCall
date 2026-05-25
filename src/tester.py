"""
Main entity hallucination testing framework with FastMCP and agent loop.

Changes vs. original:
  1. Budget-aware tool description: tracks total_num / finish_num / call_num
     and passes them to AgentConfig.get_budget_aware_description() each turn.
  2. Task-specific scoring: --task flag selects the scorer (entity / bfcl / ...).
  3. Final CSV summary: one row per sample with search_called + score columns.
"""

import asyncio
import csv
import os
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from config import AgentConfig
from agent_loop import AgentLoop
from chat_harness_agent_loop import ChatHarnessAgentLoop
from openai_agent_loop import OpenAIAgentLoop
from openai_agent_official import OpenAIOfficialAgentLoop
from tool_manager import ToolManager

# Models served via the OpenAI API (not vLLM)
_OPENAI_MODEL_PREFIXES = ("gpt-5", "gpt-4", "o1", "o3", "o4")

def _is_openai_model(model_name: str) -> bool:
    return any(model_name.lower().startswith(p) for p in _OPENAI_MODEL_PREFIXES)

def _use_official_openai_harness() -> bool:
    return os.getenv("OPENAI_AGENT_BACKEND", "chat").strip().lower() in {
        "official",
        "responses",
        "response",
    }

def _use_chat_harness() -> bool:
    return os.getenv("AGENT_HARNESS", "").strip().lower() in {
        "chat",
        "chat_template",
        "chat_harness",
    }
from mcp_client import FastMCPClient
from scorer import get_scorer, BaseScorer
from utils import load_data, ResultWriter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _search_was_called(tool_calls: list) -> bool:
    """Return True if any tool_calls entry represents an executed web_search."""
    for tc in tool_calls:
        if tc.get("type") == "tool_execution" and tc.get("tool_name") == "web_search":
            return True
        # force_search path: a selection entry with needs_tool=True counts too
        if tc.get("type") == "tool_selection":
            decision = tc.get("decision", {})
            if decision.get("needs_tool") and decision.get("tool_name") == "web_search":
                return True
    return False


def _write_csv_summary(rows: List[dict], csv_path: str) -> None:
    """Append rows to a CSV summary file, writing a header on first write."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    fieldnames = [
        "entity",
        "query",
        "model",
        "search_called",
        "yes_no_decision",
        "score",
        "correct_claims",
        "total_claims",
    ]

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main tester class
# ─────────────────────────────────────────────────────────────────────────────

class EntityHallucinationTester:
    """Main testing class for entity hallucination experiments."""

    def __init__(
        self,
        output_dir: str = "./results",
        config: AgentConfig = None,
    ):
        self.output_dir = output_dir
        self.config = config or AgentConfig()

    # ──────────────────────────────────────────────────── small helpers

    @staticmethod
    def _split_thinking_response(raw: str):
        """
        GPT-OSS models embed thinking before the final answer.
        Splits on the literal 'assistantfinal' token (case-insensitive).
        Returns (thinking, final_response).
        """
        marker = "assistantfinal"
        idx = raw.lower().find(marker)
        if idx == -1:
            return "", raw.strip()
        return raw[:idx].strip(), raw[idx + len(marker):].strip()

    def create_entity_query(self, entity: str) -> str:
        return self.config.format_entity_query(entity)

    # ──────────────────────────────────────────────── async generation loop

    async def generate_with_agent_loop(
        self,
        data: List[str],
        writer: ResultWriter,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0,
        tensor_parallel_size: int = 1,
        enable_web_search: bool = False,
        force_search: bool = False,
        search_provider: str = "brave",
        api_key: Optional[str] = None,
        save_every: int = 1,
        keep_in_memory: bool = False,
        scorer: Optional[BaseScorer] = None,
        csv_path: Optional[str] = None,
    ) -> List[dict]:

        tool_manager = ToolManager(config=self.config)

        if _is_openai_model(model_name):
            # ── OpenAI API path (GPT-5.2, GPT-4o, o3, …) ────────────────────
            if _use_official_openai_harness():
                print(f"Using OpenAI Responses API official tool path for model: {model_name}")
                agent = OpenAIOfficialAgentLoop(
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_manager=tool_manager,
                    config=self.config,
                )
            else:
                print(f"Using OpenAI API for model: {model_name}")
                agent = OpenAIAgentLoop(
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_manager=tool_manager,
                    config=self.config,
                )
        else:
            # ── vLLM path (local / self-hosted models) ────────────────────────
            from vllm import LLM, SamplingParams

            print(f"Loading model {model_name} via vLLM...")
            llm = LLM(
                model=model_name,
                max_model_len=4096,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
            )
            tokenizer = llm.get_tokenizer()
            print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.DEFAULT_TOP_P,
            )
            if _use_chat_harness():
                print("Using chat-template tool harness for local/vLLM model")
                agent = ChatHarnessAgentLoop(
                    llm=llm,
                    sampling_params=sampling_params,
                    tool_manager=tool_manager,
                    config=self.config,
                    tokenizer=tokenizer,
                )
            else:
                agent = AgentLoop(
                    llm=llm,
                    sampling_params=sampling_params,
                    tool_manager=tool_manager,
                    config=self.config,
                    tokenizer=tokenizer,
                )

        results = []
        buffer: List[dict] = []

        if enable_web_search:
            async with FastMCPClient(
                server_script=str(Path(__file__).with_name("mcp_server.py")),
                provider=search_provider,
                api_key=api_key,
            ) as mcp_client:
                tool_manager.set_mcp_client(mcp_client)
                results = await self._process_data_with_agent(
                    data=data,
                    agent=agent,
                    writer=writer,
                    model_name=model_name,
                    enable_tool_selection=not force_search,
                    force_search=force_search,
                    buffer=buffer,
                    save_every=save_every,
                    keep_in_memory=keep_in_memory,
                    scorer=scorer,
                    csv_path=csv_path,
                )
        else:
            results = await self._process_data_with_agent(
                data=data,
                agent=agent,
                writer=writer,
                model_name=model_name,
                enable_tool_selection=False,
                force_search=False,
                buffer=buffer,
                save_every=save_every,
                keep_in_memory=keep_in_memory,
                scorer=scorer,
                csv_path=csv_path,
            )

        return results

    # ──────────────────────────────────────────────── core per-sample loop

    async def _process_data_with_agent(
        self,
        data: List[str],
        agent: AgentLoop,
        writer: ResultWriter,
        model_name: str,
        enable_tool_selection: bool,
        force_search: bool,
        buffer: List[dict],
        save_every: int,
        keep_in_memory: bool,
        scorer: Optional[BaseScorer],
        csv_path: Optional[str],
    ) -> List[dict]:

        results: List[dict] = []
        total_num = len(data)

        # Running counters for the budget-aware tool description
        finish_num = 0   # number of questions fully answered before this one
        call_num = 0     # cumulative web_search executions so far

        active_key = self.config.get_active_description_key()
        use_budget_aware = self.config.is_budget_aware_key(active_key)
        use_perceived_need = self.config.is_perceived_need_key(active_key)
        use_perceived_need_v1 = self.config.is_perceived_need_v1_key(active_key)
        use_perceived_need_v2 = self.config.is_perceived_need_v2_key(active_key)

        csv_buffer: List[dict] = []

        for entity in tqdm(
            data,
            desc=f"Processing ({model_name})",
            unit="sample",
            colour="green",
        ):
            query = self.create_entity_query(entity)

            # ── 1. Build dynamic tool description (if budget-aware) ──────
            override_desc = None
            if use_budget_aware:
                override_desc = self.config.get_budget_aware_description(
                    key=active_key,
                    total_num=total_num,
                    finish_num=finish_num,
                    call_num=call_num,
                )

            # ── 2. Run agent ─────────────────────────────────────────────
            agent_response = await agent.run(
                query=query,
                enable_tool_selection=enable_tool_selection,
                force_search=force_search,
                override_description=override_desc,
                decision_only=use_budget_aware,
                perceived_need=use_perceived_need,
                perceived_need_v1=use_perceived_need_v1,
                perceived_need_v2=use_perceived_need_v2,
            )

            # ── 3. Did the model search? ──────────────────────────────────
            search_called = _search_was_called(agent_response.tool_calls)
            if search_called:
                call_num += 1

            # ── 3b. Extract YES/NO decision (actual-need variants) ───────
            yes_no_decision = None
            if use_perceived_need or use_perceived_need_v1 or use_perceived_need_v2:
                for tc in agent_response.tool_calls:
                    if tc.get("type") == "perceived_need_probe":
                        yes_no_decision = tc.get("yes_no_decision")
                        break

            # ── 4. Build JSONL row ────────────────────────────────────────
            raw_response = agent_response.response
            row = {
                "entity": entity,
                "query": query,
                "tool_in_prompt_info": agent_response.tool_in_prompt_info,
                "final_prompt": agent_response.final_prompt,
                "response": raw_response,
                "model": model_name,
                "tool_calls": agent_response.tool_calls,
                "iterations": agent_response.iterations,
                "tokens_generated": agent_response.tokens_generated,
                "web_search_enabled": enable_tool_selection,
                "search_called": search_called,
                "yes_no_decision": yes_no_decision,
            }

            # GPT-OSS: split thinking vs. final answer
            # NOTE: must happen BEFORE scoring so row["response"] holds the
            # clean final answer rather than the raw thinking+answer blob.
            if "gpt-oss" in model_name.lower():
                thinking, final_response = self._split_thinking_response(raw_response)
                row["thinking"] = thinking
                row["response"] = final_response

            # ── 5. Score ──────────────────────────────────────────────────
            score_value = None
            correct_claims = None
            total_claims = None

            # Score for ALL variants (not just "main") so every output file
            # contains factuality data.  decision_only / perceived_need runs
            # produce an empty response string and will short-circuit inside
            # the scorer (score → 0.0 / N/A), which is the correct behaviour.
            any_perceived_need = use_perceived_need or use_perceived_need_v1 or use_perceived_need_v2
            if scorer is not None and not any_perceived_need and not (use_budget_aware and not search_called):
                try:
                   # Change for other tasks
                    score_value = scorer.score(
                        entity=entity,
                        question=query,
                        response=row["response"],
                        # BFCL extras (silently ignored by entity scorer)
                        # sample_id=entity,
                        # ground_truth=row.get("ground_truth"),
                    )
                    row["score"] = score_value

                except Exception as e:
                    print(f"Warning: scoring failed for '{entity}': {e}")
                    row["score"] = None

            # ── 6. CSV summary row ────────────────────────────────────────
            if csv_path is not None:
                csv_buffer.append({
                    "entity": entity,
                    "query": query,
                    "model": model_name,
                    "search_called": search_called,
                    "yes_no_decision": yes_no_decision,
                    "score": score_value,
                    "correct_claims": correct_claims,
                    "total_claims": total_claims,
                })

            # ── 7. Buffer / checkpoint ────────────────────────────────────
            buffer.append(row)
            if keep_in_memory:
                results.append(row)

            if len(buffer) >= save_every:
                for r in buffer:
                    writer.write_one(r)
                buffer.clear()

                if csv_path and csv_buffer:
                    _write_csv_summary(csv_buffer, csv_path)
                    csv_buffer.clear()

            finish_num += 1

        # Final flush
        if buffer:
            for r in buffer:
                writer.write_one(r)
            buffer.clear()

        if csv_path and csv_buffer:
            _write_csv_summary(csv_buffer, csv_path)
            csv_buffer.clear()

        return results if keep_in_memory else []

    # ──────────────────────────────────────────────── public synchronous API

    def run_test(
        self,
        data_file: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0,
        tensor_parallel_size: int = 1,
        enable_web_search: bool = False,
        force_search: bool = False,
        search_provider: str = "brave",
        api_key: Optional[str] = None,
        save_every: int = 1,
        keep_in_memory: bool = False,
        limit: Optional[int] = None,
        task: str = "entity",
        scorer_kwargs: Optional[dict] = None,
        skip_scorer: bool = False,
    ) -> List[dict]:
        """
        Run the complete test pipeline.

        New args vs. original:
            task:           Scorer task ("entity", "bfcl", …).
            scorer_kwargs:  Extra kwargs forwarded to the scorer constructor.
            skip_scorer:    If True, skip scorer instantiation and scoring entirely.
        """
        # Load data
        print(f"Loading data from {data_file}...")
        df = load_data(data_file)
        data = df["entity_text"].tolist()

        if limit:
            data = data[:limit]
            print(f"Limited to {limit} samples")

        print(f"Loaded {len(data)} samples")

        # Adjust max_tokens for specific model families
        max_tokens = self.config.get_max_tokens_for_model(model_name)

        # Build output base name
        if force_search:
            search_suffix = "force_search"
        else:
            search_suffix = "with_search" if enable_web_search else "no_search"

        safe_model = model_name.replace("/", "_")
        # Strip the long cluster storage prefix that appears in absolute model paths,
        # e.g. "_NS_factual-knowledge-and-hallucination_nobackup_qwu_llm_base_model_"
        # so output filenames stay readable.
        import re as _re
        safe_model = _re.sub(
            r"_NS_factual-knowledge-and-hallucination_nobackup_qwu_llm_base_model_",
            "",
            safe_model,
        )
        base_name = f"vllm_{safe_model}_{search_suffix}"
        if _is_openai_model(model_name) and _use_official_openai_harness():
            base_name = f"{base_name}_official_harness"
        elif not _is_openai_model(model_name) and _use_chat_harness():
            base_name = f"{base_name}_chat_harness"

        csv_path = os.path.join(self.output_dir, f"{base_name}_summary.csv")

        # Instantiate scorer
        scorer_kwargs = scorer_kwargs or {}
        if skip_scorer:
            scorer = None
            print("Scorer: skipped (--skip-scorer)")
        else:
            try:
                scorer = get_scorer(task, **scorer_kwargs)
                print(f"Scorer task: {task} → {scorer.__class__.__name__}")
            except Exception as e:
                print(f"Warning: could not initialise scorer for task '{task}': {e}")
                scorer = None

        with ResultWriter(output_dir=self.output_dir, base_name=base_name) as writer:
            print(f"Streaming JSONL to : {writer.jsonl_path}")
            print(f"CSV summary to     : {csv_path}\n")

            results = asyncio.run(
                self.generate_with_agent_loop(
                    data=data,
                    writer=writer,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tensor_parallel_size=tensor_parallel_size,
                    enable_web_search=enable_web_search,
                    force_search=force_search,
                    search_provider=search_provider,
                    api_key=api_key,
                    save_every=save_every,
                    keep_in_memory=keep_in_memory,
                    scorer=scorer,
                    csv_path=csv_path
                )
            )

        print("\nCompleted! Results saved incrementally.")
        print(f"CSV summary: {csv_path}")
        return results
