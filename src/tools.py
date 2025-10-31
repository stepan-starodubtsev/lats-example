import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

load_dotenv()

tavily_tool = TavilySearch(tavily_api_key=SecretStr(os.getenv("TAVILY_API_KEY")))
tools = [tavily_tool]
tool_node = ToolNode(tools=tools)
