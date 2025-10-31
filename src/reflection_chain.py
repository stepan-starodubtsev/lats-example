from langchain_core.messages import AIMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable

from src.llm_setup import llm
from src.reflection import Reflection


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a harsh critic. Reflect and grade the assistant response to the user question below.\n"
            "Be very strict with your scoring. A score of 5-6 is for an average response. A score of 10 is reserved only for perfect, flawless, and comprehensive responses that fully satisfy all aspects of the user's request.\n"
            "Penalize heavily for any inaccuracies, lack of detail, or failure to follow instructions.\n"
            "Consider it a valid final answer only if the last message is an assistant\n"
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_llm_chain = (
    prompt
    | llm.bind_tools(tools=[Reflection],
                     tool_choice="Reflection",
                     parallel_tool_calls=False).with_config(
        run_name="Reflection"
    )
    | PydanticToolsParser(tools=[Reflection])
)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    MIN_SCORE_THRESHOLD = 9

    tool_choices = reflection_llm_chain.invoke(inputs)
    reflection = tool_choices[0]

    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False

    if reflection.score < MIN_SCORE_THRESHOLD:
        reflection.found_solution = False
    else:
        reflection.found_solution = True

    return reflection


