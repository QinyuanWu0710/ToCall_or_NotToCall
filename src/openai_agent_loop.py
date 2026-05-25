"""
OpenAI-native agent loop for GPT-5.x models (gpt-5.2, gpt-5.2-chat-latest, etc.).

Drop-in replacement for AgentLoop when the model is served via the OpenAI API
rather than vLLM.  The public interface (`await agent.run(...)`) is identical so
tester.py / _process_data_with_agent needs no changes.
"""

from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

import openai  # pip install openai>=1.0

from config import AgentConfig
# Re-use the same data-classes as the vLLM loop so the rest of the pipeline
# (tester.py, scorer.py, …) keeps working without any changes.
from agent_loop import AgentResponse, ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Models that accept a `reasoning` parameter (effort-based thinking)
_REASONING_MODELS = {
    "gpt-5.5",
}

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI agent loop
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIAgentLoop:
    """
    Agent loop that mirrors AgentLoop but calls the OpenAI Chat Completions API.

    Parameters
    ----------
    model_name : str
        OpenAI model identifier, e.g. ``"gpt-5.2"`` or ``"gpt-5.2-chat-latest"``.
    max_tokens : int
        Maximum output tokens per generation call.
    temperature : float
        Sampling temperature 
    tool_manager : ToolManager
        Shared tool manager (same as used by the vLLM loop).
    config : AgentConfig | None
        Shared agent config.
    api_key : str | None
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` env-var.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        tool_manager: Any = None,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_manager = tool_manager
        self.config = config or AgentConfig()
        self._client = openai.AsyncOpenAI(api_key=api_key)  # key from env if None

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_kwargs(self) -> Dict[str, Any]:
        """Build extra keyword arguments for the API call."""
        kwargs: Dict[str, Any] = {"max_completion_tokens": self.max_tokens}
        kwargs["temperature"] = self.temperature

        return kwargs

    async def _chat(self, messages: List[Dict[str, str]]) -> tuple[str, int]:
        """
        Call the OpenAI Chat Completions API.

        Returns
        -------
        (text, token_count)
        """
        extra = self._build_kwargs()
        resp = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **extra,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.completion_tokens if resp.usage else 0
        return text, tokens

    def _parse_tool_decision(self, response: str) -> ToolCall:
        """Parse JSON tool-selection output (same logic as AgentLoop)."""
        try:
            if "assistantfinal" in response:
                response = response.split("assistantfinal", 1)[1]
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return ToolCall(
                    needs_tool=data.get("needs_tool", False),
                    tool_name=data.get("tool_name"),
                    tool_input=data.get("tool_input", ""),
                    reasoning=data.get("reasoning", ""),
                )
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool decision: {e}\nResponse: {response}")
        return ToolCall(needs_tool=False, tool_name=None, tool_input=None, reasoning="parse error")

    def _format_search_results(self, raw: Any) -> str:
        """Format MCP web-search results for injection into the final prompt."""
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if "error" in data:
                return f"Search error: {data['error']}"
            results = data.get("results", [])
            if not results:
                return "No search results found."
            lines = ["Search Results:\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', '')}")
                lines.append(f"   {r.get('snippet', '')}")
                if r.get("link"):
                    lines.append(f"   Source: {r['link']}")
                lines.append("")
            return "\n".join(lines).strip()
        except Exception as e:
            return f"Error formatting search results: {e}"

    # ── public API (mirrors AgentLoop.run) ────────────────────────────────────

    async def run(
        self,
        query: str,
        enable_tool_selection: bool = True,
        force_search: bool = False,
        override_description: Optional[str] = None,
        decision_only: bool = False,
        perceived_need: bool = False,
        perceived_need_v1: bool = False,
        perceived_need_v2: bool = False
    ) -> AgentResponse:
        """
        Run the agent loop for a single query.  Signature is identical to
        ``AgentLoop.run`` so tester.py needs no changes.
        """
        tool_calls: List[Dict[str, Any]] = []
        iterations = 0
        total_tokens = 0
        tool_in_prompt_info = ""

        # ── perceived-need probe ──────────────────────────────────────────────
        if perceived_need:
            iterations += 1
            content = (
                f"{query}\n\nYou can call for help from a tool.\n\n"
                "Respond with ONLY a JSON object in this exact schema:\n"
                '{\n  "needs_tool": true\n}\nor\n{\n  "needs_tool": false\n}\n'
            )
            messages = [{"role": "user", "content": content}]
            raw, toks = await self._chat(messages)
            total_tokens += toks
            parsed = self._parse_tool_decision(raw)
            tool_calls.append({
                "type": "perceived_need_probe",
                "prompt": content,
                "raw_response": raw,
                "yes_no_decision": parsed.needs_tool,
            })
            return AgentResponse(
                tool_in_prompt_info=content,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )

        tool_decision: Optional[ToolCall] = None

        # ── force-search path ─────────────────────────────────────────────────
        if force_search and self.tool_manager and self.tool_manager.has_tools():
            iterations += 1
            content = self.config.format_tool_force_prompt(query, override_description=override_description)
            messages = [{"role": "user", "content": content}]
            raw, toks = await self._chat(messages)
            total_tokens += toks
            parsed = self._parse_tool_decision(raw)
            tool_decision = ToolCall(
                needs_tool=True,
                tool_name="web_search",
                tool_input=(parsed.tool_input or "").strip() or query,
                reasoning=parsed.reasoning or "Force search enabled",
            )

        # ── normal tool-selection path ────────────────────────────────────────
        if not force_search and enable_tool_selection and self.tool_manager and self.tool_manager.has_tools():
            iterations += 1
            content = self.config.format_tool_selection_prompt(query, override_description=override_description)
            tool_in_prompt_info = content
            messages = [{"role": "user", "content": content}]
            raw, toks = await self._chat(messages)
            total_tokens += toks
            tool_decision = self._parse_tool_decision(raw)
            tool_calls.append({
                "type": "tool_selection",
                "decision": {
                    "needs_tool": tool_decision.needs_tool,
                    "tool_name": tool_decision.tool_name,
                    "tool_input": tool_decision.tool_input,
                    "reasoning": tool_decision.reasoning,
                },
                "raw_response": raw,
            })

        # ── decision-only mode ────────────────────────────────────────────────
        if decision_only:
            return AgentResponse(
                tool_in_prompt_info=tool_in_prompt_info,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )

        # ── execute tool if needed ────────────────────────────────────────────
        search_context = ""
        if tool_decision and tool_decision.needs_tool and tool_decision.tool_name:
            iterations += 1
            tool_input = (tool_decision.tool_input or "").strip() or query
            tool_result = await self.tool_manager.execute_tool(
                tool_decision.tool_name, query=tool_input
            )
            tool_calls.append({
                "type": "tool_execution",
                "tool_name": tool_decision.tool_name,
                "tool_input": tool_decision.tool_input,
                "result": tool_result,
            })
            if tool_decision.tool_name == "web_search":
                search_context = self._format_search_results(tool_result)

        # ── final response ────────────────────────────────────────────────────
        iterations += 1
        if search_context:
            final_content = (
                f"Based on the following search results, please answer the question.\n\n"
                f"{search_context}\n\nQuestion: {query}"
            )
        else:
            final_content = query

        messages = [{"role": "user", "content": final_content}]
        system_msg = self.config.get_system_message(model_name=self.model_name)
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})

        final_response, toks = await self._chat(messages)
        total_tokens += toks

        return AgentResponse(
            tool_in_prompt_info=tool_in_prompt_info,
            final_prompt=final_content,
            response=final_response,
            tool_calls=tool_calls,
            iterations=iterations,
            tokens_generated=total_tokens,
        )

    def run_sync(self, query: str, enable_tool_selection: bool = True) -> AgentResponse:
        """Synchronous convenience wrapper."""
        return asyncio.run(self.run(query, enable_tool_selection))