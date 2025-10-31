from src.config import _set_if_undefined  # ensure env vars present
from src.graph_build import build_graph


def main():
    # Ensure keys are set (prompts if missing in env)
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    graph = build_graph()

    question = (
        "Generate a table with the average size and weight, as well as the oldest "
        "recorded instance for each of the top 5 most common birds."
    )

    last_step = None
    for step in graph.stream({"input": question}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")

    solution_node = last_step["expand"]["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    print(best_trajectory[-1].content)


if __name__ == "__main__":
    main()