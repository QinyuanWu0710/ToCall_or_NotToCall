"""
Chat-template tool harness for local/vLLM models.

This variant keeps the same public interface as AgentLoop, but it formats tool
availability, assistant tool calls, and tool results through the tokenizer chat
template instead of flattening search results into a plain user prompt.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

from agent_loop import AgentResponse, AgentLoop, ToolCall
from config import AgentConfig


class ChatHarnessAgentLoop(AgentLoop):
    """
    Agent loop that uses chat-template-native tool messages.

    The decision step passes a function schema via apply_chat_template(...,
    tools=[...]). If the model asks for web_search, final generation receives
    the full chat history:
      system -> user -> assistant(tool_call) -> tool(result) -> assistant
    """

    def __init__(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        tool_manager: Any,
        config: Optional[AgentConfig] = None,
        tokenizer: Any = None,
    ):
        super().__init__(
            llm=llm,
            sampling_params=sampling_params,
            tool_manager=tool_manager,
            config=config,
            tokenizer=tokenizer,
        )

    def _tool_schema(self, override_description: Optional[str] = None) -> Dict[str, Any]:
        tool = self.config.get_tool_by_name("web_search", override_description)
        description = (
            tool.description
            if tool is not None
            else "Search the web for current information about entities, facts, or topics."
        )
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A short, effective web search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def _system_message(self) -> Optional[Dict[str, str]]:
        system_msg = self.config.get_system_message(model_name=self.model_name)
        if system_msg:
            return {"role": "system", "content": system_msg}
        return None

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if self.tokenizer is None:
            formatted = ""
            for msg in messages:
                formatted += f"{msg['role']}: {msg.get('content', '')}\n"
            return formatted

        try:
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if tools is not None:
                kwargs["tools"] = tools
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to apply chat template with tools={tools is not None}: {e}")
            formatted = ""
            if tools is not None:
                formatted += f"tools: {json.dumps(tools, ensure_ascii=False)}\n"
            for msg in messages:
                formatted += f"{msg['role']}: {msg.get('content', '')}\n"
                if msg.get("tool_calls"):
                    formatted += f"tool_calls: {json.dumps(msg['tool_calls'], ensure_ascii=False)}\n"
            return formatted

    def _decision_messages(
        self,
        query: str,
        *,
        force_search: bool = False,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        system_msg = self._system_message()
        if system_msg:
            messages.append(system_msg)

        if force_search:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{query}\n\n"
                        "You must call the web_search tool. Use a short, effective "
                        "search query for the tool arguments."
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": query})
        return messages

    def _create_tool_selection_prompt(self, query: str, override_description: str = None) -> str:
        messages = self._decision_messages(query, force_search=False)
        return self._apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=[self._tool_schema(override_description)],
        )

    def _create_force_search_prompt(self, query: str, override_description: str = None) -> str:
        messages = self._decision_messages(query, force_search=True)
        return self._apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=[self._tool_schema(override_description)],
        )

    @staticmethod
    def _extract_tool_call_json(response: str) -> Optional[Dict[str, Any]]:
        if "<tool_call>" in response and "</tool_call>" in response:
            start = response.find("<tool_call>") + len("<tool_call>")
            end = response.find("</tool_call>", start)
            candidate = response[start:end].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                return None
        return None

    def _parse_tool_decision(self, response: str) -> ToolCall:
        if "assistantfinal" in response:
            response = response.split("assistantfinal", 1)[1]
        if "\n</think>\n" in response:
            response = response.split("\n</think>\n", 1)[1]

        data = self._extract_tool_call_json(response)
        if data is None:
            return ToolCall(
                needs_tool=False,
                tool_name=None,
                tool_input=None,
                reasoning="No valid tool call response",
            )

        if "function" in data:
            function = data.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = data.get("name") or data.get("tool_name")
            arguments = data.get("arguments", data)

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"query": arguments}

        if not isinstance(arguments, dict):
            arguments = {}

        tool_input = (
            arguments.get("query")
            or arguments.get("tool_input")
            or data.get("tool_input")
            or ""
        )
        needs_tool = bool(name) or bool(data.get("needs_tool", False))
        tool_name = name or ("web_search" if needs_tool else None)

        return ToolCall(
            needs_tool=needs_tool,
            tool_name=tool_name,
            tool_input=tool_input,
            reasoning=data.get("reasoning", "Chat-template tool call decision"),
        )

    def _assistant_tool_call_message(self, tool_name: str, tool_input: str) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": {"query": tool_input},
                    },
                }
            ],
        }

    def _final_messages(
        self,
        query: str,
        *,
        tool_decision: Optional[ToolCall],
        search_context: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        system_msg = self._system_message()
        if system_msg:
            messages.append(system_msg)

        messages.append({"role": "user", "content": query})

        if search_context and tool_decision and tool_decision.tool_name:
            tool_input = (tool_decision.tool_input or "").strip() or query
            messages.append(self._assistant_tool_call_message(tool_decision.tool_name, tool_input))
            messages.append({"role": "tool", "content": search_context})

        return messages

    async def run(
        self,
        query: str,
        enable_tool_selection: bool = True,
        force_search: bool = False,
        override_description: str = None,
        decision_only: bool = False,
        perceived_need: bool = False,
        perceived_need_v1: bool = False,
        perceived_need_v2: bool = False,
    ) -> AgentResponse:
        tool_calls: List[Dict[str, Any]] = []
        iterations = 0
        total_tokens = 0
        tool_decision: Optional[ToolCall] = None
        tool_in_prompt_info = ""

        if perceived_need or perceived_need_v1 or perceived_need_v2:
            return await super().run(
                query=query,
                enable_tool_selection=enable_tool_selection,
                force_search=force_search,
                override_description=override_description,
                decision_only=decision_only,
                perceived_need=perceived_need,
                perceived_need_v1=perceived_need_v1,
                perceived_need_v2=perceived_need_v2,
            )

        if force_search and self.tool_manager.has_tools():
            iterations += 1
            force_prompt = self._create_force_search_prompt(query, override_description)
            outputs = self.llm.generate([force_prompt], self.sampling_params)
            decision_response = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)
            parsed = self._parse_tool_decision(decision_response)
            tool_decision = ToolCall(
                needs_tool=True,
                tool_name="web_search",
                tool_input=(parsed.tool_input or "").strip() or query,
                reasoning=parsed.reasoning or "Force search enabled",
            )
            tool_calls.append(
                {
                    "type": "tool_selection",
                    "decision": {
                        "needs_tool": tool_decision.needs_tool,
                        "tool_name": tool_decision.tool_name,
                        "tool_input": tool_decision.tool_input,
                        "reasoning": tool_decision.reasoning,
                    },
                    "raw_response": decision_response,
                }
            )

        if not force_search and enable_tool_selection and self.tool_manager.has_tools():
            iterations += 1
            tool_selection_prompt = self._create_tool_selection_prompt(query, override_description)
            tool_in_prompt_info = tool_selection_prompt
            outputs = self.llm.generate([tool_selection_prompt], self.sampling_params)
            decision_response = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)
            tool_decision = self._parse_tool_decision(decision_response)
            tool_calls.append(
                {
                    "type": "tool_selection",
                    "decision": {
                        "needs_tool": tool_decision.needs_tool,
                        "tool_name": tool_decision.tool_name,
                        "tool_input": tool_decision.tool_input,
                        "reasoning": tool_decision.reasoning,
                    },
                    "raw_response": decision_response,
                }
            )

        if decision_only:
            return AgentResponse(
                tool_in_prompt_info=tool_in_prompt_info,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )

        search_context = ""
        if tool_decision and tool_decision.needs_tool and tool_decision.tool_name:
            iterations += 1
            tool_input = (tool_decision.tool_input or "").strip() or query
            tool_result = await self.tool_manager.execute_tool(
                tool_decision.tool_name,
                query=tool_input,
            )
            tool_calls.append(
                {
                    "type": "tool_execution",
                    "tool_name": tool_decision.tool_name,
                    "tool_input": tool_input,
                    "result": tool_result,
                }
            )
            if tool_decision.tool_name == "web_search":
                search_context = self._format_search_results(tool_result)

        iterations += 1
        final_messages = self._final_messages(
            query,
            tool_decision=tool_decision,
            search_context=search_context,
        )
        final_prompt = self._apply_chat_template(
            final_messages,
            add_generation_prompt=True,
            tools=[self._tool_schema(override_description)] if search_context else None,
        )
        outputs = self.llm.generate([final_prompt], self.sampling_params)
        final_response = outputs[0].outputs[0].text
        total_tokens += len(outputs[0].outputs[0].token_ids)

        return AgentResponse(
            tool_in_prompt_info=tool_in_prompt_info,
            final_prompt=final_prompt,
            response=final_response,
            tool_calls=tool_calls,
            iterations=iterations,
            tokens_generated=total_tokens,
        )

    def run_sync(self, query: str, enable_tool_selection: bool = True) -> AgentResponse:
        return asyncio.run(self.run(query, enable_tool_selection))

