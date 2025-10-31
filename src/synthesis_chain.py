from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable

from src.llm_setup import llm


# Prompt to force a single synthesized answer after tool outputs
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You must produce a concise final assistant message that:\n"
            "- Selects exactly one most innovative feature.\n"
            "- Provides one short paragraph justification.\n"
            "- Includes 1-3 explicit URLs used as evidence.\n"
            "- Do not include tool-call JSON, only natural language.\n",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


@as_runnable
def synthesis_chain(inputs):
    return (prompt | llm.with_config(run_name="SynthesizeAnswer")).invoke(inputs)


