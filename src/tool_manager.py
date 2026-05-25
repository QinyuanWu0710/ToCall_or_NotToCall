"""
Tool manager for registering and executing tools.
Handles the interface between the agent and MCP tools.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from config import AgentConfig, ToolConfig


@dataclass
class Tool:
    """Represents a registered tool."""
    name: str
    description: str
    executor: Callable
    is_async: bool = True


class ToolManager:
    """
    Manages available tools and their execution.
    """
    
    def __init__(self, config: AgentConfig = None):
        """
        Initialize the tool manager.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.tools: Dict[str, Tool] = {}
        self._mcp_client = None
    
    def register_tool(
        self,
        name: str,
        description: str,
        executor: Callable,
        is_async: bool = True
    ):
        """
        Register a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            executor: Function to execute the tool
            is_async: Whether the executor is async
        """
        self.tools[name] = Tool(
            name=name,
            description=description,
            executor=executor,
            is_async=is_async
        )
    
    def set_mcp_client(self, mcp_client):
        """
        Set the MCP client for web search.
        
        Args:
            mcp_client: FastMCP client instance
        """
        self._mcp_client = mcp_client
        
        # Register web search tool
        if self._mcp_client:
            self.register_tool(
                name="web_search",
                description="Search the web for current information",
                executor=self._execute_web_search,
                is_async=True
            )
    
    async def _execute_web_search(self, query: str, count: int = 5) -> str:
        """Execute web search via MCP."""
        if not self._mcp_client:
            return '{"error": "MCP client not initialized"}'
        
        try:
            # Call the web_search tool via MCP
            result = await self._mcp_client.call_tool(
                "web_search",
                arguments={"query": query, "count": count}
            )
            
            # Return the result (should be JSON string)
            if isinstance(result, str):
                return result
            elif hasattr(result, 'content'):
                # Handle MCP response format
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    return content[0].text if hasattr(content[0], "text") else "{}"
                return '{}'
            else:
                return str(result)
        
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        try:
            if tool.is_async:
                result = await tool.executor(**kwargs)
            else:
                result = tool.executor(**kwargs)
            print(f'executor results: {result}')
            return result
        
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def has_tools(self) -> bool:
        """Check if any tools are registered."""
        return len(self.tools) > 0
    
    def get_tool_list(self) -> List[ToolConfig]:
        """Get list of registered tools as ToolConfig objects."""
        return [
            ToolConfig(
                name=tool.name,
                description=tool.description,
                enabled=True
            )
            for tool in self.tools.values()
        ]
    
    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.tools.keys())