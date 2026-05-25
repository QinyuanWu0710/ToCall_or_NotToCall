"""
OpenAI Responses API agent loop.

This is a drop-in alternative to openai_agent_loop.OpenAIAgentLoop for running
the same entity hallucination experiment while letting the OpenAI Responses API
make the web-search decision through its official tool interface.

The public interface mirrors AgentLoop.run(...) and returns the same
AgentResponse / tool_calls shape expected by tester.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI

from agent_loop import AgentResponse, ToolCall
from config import AgentConfig


MCP_RESPONSES_URL_BACKENDS = {
    "mcp",
    "mcp_url",
    "response_url",
    "responses_url",
    "openai_responses_url",
}

MCP_PROVIDER_ALIASES = {
    "google": "mcp-serp",
    "serp": "mcp-serp",
    "serpapi": "mcp-serp",
    "mcp_serp": "mcp-serp",
    "mcp-serp": "mcp-serp",
    "perplexity": "mcp-perplexity",
    "mcp_perplexity": "mcp-perplexity",
    "mcp-perplexity": "mcp-perplexity",
    "brave": "mcp-brave",
    "mcp_brave": "mcp-brave",
    "mcp-brave": "mcp-brave",
    "tavily": "mcp-tavily",
    "mcp_tavily": "mcp-tavily",
    "mcp-tavily": "mcp-tavily",
}


def _csv_env(name: str) -> Optional[List[str]]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_mcp_provider(provider: Optional[str]) -> str:
    key = (provider or "mcp-perplexity").strip().lower()
    return MCP_PROVIDER_ALIASES.get(key, key)


def _mcp_server_url(provider: str) -> Optional[str]:
    response_url = os.getenv("MCP_RESPONSE_URL", "").strip()
    if response_url:
        return response_url

    explicit_url = os.getenv("MCP_SERVER_URL", "").strip()
    if explicit_url:
        return explicit_url

    if provider == "mcp-serp":
        api_key = os.getenv("SERPAPI_API_KEY", "").strip()
        if api_key:
            return f"https://mcp.serpapi.com/{api_key}/mcp"

    return None


def _mcp_tool_config(provider: str) -> Dict[str, Any]:
    server_url = _mcp_server_url(provider)
    if not server_url:
        raise ValueError(
            "MCP_SERVER_URL must be set for MCP Responses backends, unless "
            "MCP_SEARCH_PROVIDER=mcp-serp and SERPAPI_API_KEY is available."
        )

    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": os.getenv("MCP_SERVER_LABEL", provider.replace("-", "_")),
        "server_url": server_url,
        "require_approval": os.getenv("MCP_REQUIRE_APPROVAL", "never"),
    }
    allowed_tools = _csv_env("MCP_ALLOWED_TOOLS")
    if allowed_tools is not None:
        tool["allowed_tools"] = allowed_tools
    authorization = os.getenv("MCP_AUTHORIZATION", "").strip()
    if authorization:
        tool["authorization"] = authorization
    return tool


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(mode="json", warnings=False)
        except TypeError:
            return response.model_dump()
    if isinstance(response, dict):
        return response
    return {"repr": repr(response)}


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            part = getattr(content, "text", None)
            if isinstance(part, str):
                chunks.append(part)
    return "\n".join(chunks).strip()


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    for attr in ("output_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            return value
    if isinstance(usage, dict):
        for key in ("output_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
    return 0


def _walk_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _best_query_from_event(event: Dict[str, Any]) -> Optional[str]:
    action = event.get("action")
    if isinstance(action, dict):
        for key in ("query", "search_query"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    arguments = event.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if isinstance(arguments, dict):
        for key in ("query", "search_query", "q"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("query", "search_query"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


class OpenAIOfficialAgentLoop:
    """
    Agent loop that uses OpenAI Responses API tools for the search decision.

    Tool backend is controlled by OPENAI_AGENT_TOOL_BACKEND, falling back to
    REPLAY_TOOL_BACKEND and then openai_web:
      - openai_web: use OpenAI's hosted web_search tool
      - responses_url / mcp_url / mcp: use a remote MCP server URL
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        tool_manager: Any = None,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_manager = tool_manager
        self.config = config or AgentConfig()
        self._client = AsyncOpenAI(api_key=api_key)

    def _tool_backend(self) -> str:
        return os.getenv(
            "OPENAI_AGENT_TOOL_BACKEND",
            os.getenv("REPLAY_TOOL_BACKEND", "openai_web"),
        ).strip().lower()

    def _responses_input(
        self,
        query: str,
        *,
        search_context: bool,
        force_search: bool,
        override_description: Optional[str],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        system_msg = self.config.get_system_message(model_name=self.model_name)
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        if search_context:
            if force_search:
                policy = self.config.format_tool_force_prompt(
                    query, override_description=override_description
                )
            else:
                tool_list = self.config.get_tool_list_string(override_description)
                policy = (
                    "You may use the available web search tool when it is useful.\n\n"
                    f"Available tools:\n{tool_list}\n\n"
                    "Follow the tool description and cost/budget instructions when "
                    "deciding whether to search. If you search, answer the user's "
                    "question after the tool result is available. If you do not "
                    "search, answer from your own knowledge."
                )
            messages.append({"role": "system", "content": policy})

        messages.append({"role": "user", "content": query})
        return messages

    def _tool_config(self) -> Dict[str, Any]:
        backend = self._tool_backend()
        if backend in MCP_RESPONSES_URL_BACKENDS:
            provider = _normalize_mcp_provider(os.getenv("MCP_SEARCH_PROVIDER"))
            return _mcp_tool_config(provider)
        if backend in {"openai_web", "web_search", "openai"}:
            return {"type": "web_search"}
        raise ValueError(
            f"Unsupported OPENAI_AGENT_TOOL_BACKEND={backend!r}. "
            "Use openai_web or responses_url."
        )

    def _response_kwargs(
        self,
        query: str,
        *,
        enable_tools: bool,
        force_search: bool,
        override_description: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": self._responses_input(
                query,
                search_context=enable_tools,
                force_search=force_search,
                override_description=override_description,
            ),
            "max_output_tokens": self.max_tokens,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if enable_tools:
            kwargs["tools"] = [self._tool_config()]
            kwargs["tool_choice"] = "required" if force_search else "auto"
            if self._tool_backend() in {"openai_web", "web_search", "openai"}:
                kwargs["include"] = ["web_search_call.action.sources"]

        return kwargs

    async def _responses_create(self, **kwargs: Any) -> Any:
        return await self._client.responses.create(**kwargs)

    def _parse_tool_decision(self, response: str) -> ToolCall:
        try:
            if "assistantfinal" in response:
                response = response.split("assistantfinal", 1)[1]
            if "\n</think>\n" in response:
                response = response.split("\n</think>\n", 1)[1]
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return ToolCall(
                    needs_tool=bool(data.get("needs_tool", False)),
                    tool_name=data.get("tool_name"),
                    tool_input=data.get("tool_input", ""),
                    reasoning=data.get("reasoning", ""),
                )
        except json.JSONDecodeError as exc:
            print(f"Warning: Failed to parse tool decision: {exc}")
            print(f"Response was: {response}")
        return ToolCall(
            needs_tool=False,
            tool_name=None,
            tool_input=None,
            reasoning="No valid JSON response",
        )

    async def _perceived_need_probe(self, query: str, prompt_style: str) -> AgentResponse:
        if prompt_style == "v1":
            content = f"{query}\n\nDo you need help to answer the question?\n\nAnswer:"
        elif prompt_style == "v2":
            content = f"{query}\n\nDo you know the answer of the tool?\nAnswer:"
        else:
            content = (
                f"{query}\n\n"
                "You can call for help from a tool.\n\n"
                "Respond with ONLY a JSON object in this exact schema:\n"
                '{\n  "needs_tool": true\n}\n'
                "or\n"
                '{\n  "needs_tool": false\n}\n'
            )

        response = await self._responses_create(
            model=self.model_name,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self.max_tokens,
        )
        raw = _extract_text(response)
        parsed = self._parse_tool_decision(raw)
        return AgentResponse(
            tool_in_prompt_info=content,
            final_prompt="",
            response="",
            tool_calls=[
                {
                    "type": "perceived_need_probe",
                    "prompt": content,
                    "raw_response": raw,
                    "raw_response_payload": _response_to_dict(response),
                    "yes_no_decision": parsed.needs_tool,
                }
            ],
            iterations=1,
            tokens_generated=_usage_tokens(response),
        )

    def _tool_events(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for item in _walk_dicts(payload.get("output", [])):
            item_type = item.get("type")
            if item_type in {"web_search_call", "mcp_call"}:
                events.append(item)
        return events

    def _tool_calls_from_response(
        self,
        response: Any,
        *,
        query: str,
        raw_prompt: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        payload = _response_to_dict(response)
        events = self._tool_events(payload)
        first_query = None
        if events:
            first_query = _best_query_from_event(events[0])
        tool_input = first_query or query
        needs_tool = bool(events)

        calls: List[Dict[str, Any]] = [
            {
                "type": "tool_selection",
                "decision": {
                    "needs_tool": needs_tool,
                    "tool_name": "web_search" if needs_tool else None,
                    "tool_input": tool_input if needs_tool else None,
                    "reasoning": "Responses API tool_choice decision",
                },
                "raw_response": _extract_text(response),
                "raw_response_payload": payload,
                "prompt": raw_prompt,
            }
        ]

        for event in events:
            calls.append(
                {
                    "type": "tool_execution",
                    "tool_name": "web_search",
                    "tool_input": _best_query_from_event(event) or tool_input,
                    "result": event,
                }
            )

        return calls

    async def run(
        self,
        query: str,
        enable_tool_selection: bool = True,
        force_search: bool = False,
        override_description: Optional[str] = None,
        decision_only: bool = False,
        perceived_need: bool = False,
        perceived_need_v1: bool = False,
        perceived_need_v2: bool = False,
    ) -> AgentResponse:
        if perceived_need or perceived_need_v1 or perceived_need_v2:
            prompt_style = "v2" if perceived_need_v2 else "v1" if perceived_need_v1 else "main"
            return await self._perceived_need_probe(query, prompt_style)

        tools_available = bool(self.tool_manager and self.tool_manager.has_tools())
        enable_tools = bool((force_search or enable_tool_selection) and tools_available)
        kwargs = self._response_kwargs(
            query,
            enable_tools=enable_tools,
            force_search=force_search,
            override_description=override_description,
        )
        response = await self._responses_create(**kwargs)
        prompt = kwargs["input"]
        tokens = _usage_tokens(response)
        final_response = _extract_text(response)

        if enable_tools:
            tool_calls = self._tool_calls_from_response(
                response,
                query=query,
                raw_prompt=prompt,
            )
            tool_in_prompt_info = json.dumps(prompt, ensure_ascii=False, indent=2)
        else:
            tool_calls = []
            tool_in_prompt_info = ""

        return AgentResponse(
            tool_in_prompt_info=tool_in_prompt_info,
            final_prompt="" if decision_only else json.dumps(prompt, ensure_ascii=False, indent=2),
            response="" if decision_only else final_response,
            tool_calls=tool_calls,
            iterations=1,
            tokens_generated=tokens,
        )

    def run_sync(self, query: str, enable_tool_selection: bool = True) -> AgentResponse:
        return asyncio.run(self.run(query, enable_tool_selection))
