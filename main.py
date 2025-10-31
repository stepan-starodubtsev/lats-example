import time

from dotenv import load_dotenv

from src.config import set_env_if_undefined, langfuse_handler  # ensure env vars present
from src.graph_build import build_graph

load_dotenv()


def main():
    # Ensure keys are set (prompts if missing in env)
    set_env_if_undefined("OPENAI_API_KEY")
    set_env_if_undefined("TAVILY_API_KEY")

    graph = build_graph()

    question = (
        "Generate a table with the average size and weight, as well as the oldest "
        "recorded instance for each of the top 5 most common birds."
    )

    start_time = time.time()
    last_step = None
    for step in graph.stream({"input": question},
                             config={"callbacks": [langfuse_handler]}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")

    end_time = time.time()
    elapsed_time = end_time - start_time

    solution_node = last_step["expand"]["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    print(best_trajectory[-1].content)

    print(f"Method finished in: {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    main()
