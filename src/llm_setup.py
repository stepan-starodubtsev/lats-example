import os

from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.tools import tools

load_dotenv()

# Base LLM
llm = ChatOpenAI(api_key=SecretStr("EMPTY"),
                 base_url=f"{os.getenv('CHAT_API_URL')}:{os.getenv('CHAT_API_PORT')}/v1",
                 model=os.getenv("CHAT_API_MODEL"))

# Prompt used for both initial response and expansion
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

# Chain to generate the first response
initial_answer_chain = (prompt_template
                        | llm.bind_tools(tools=tools,
                                         tool_choice="required",
                                         parallel_tool_calls=False,
                                         ).with_config(run_name="GenerateInitialCandidate"))

# Parser for tool-calls from model
parser = JsonOutputToolsParser(return_id=True)
