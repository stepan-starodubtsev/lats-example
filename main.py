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
        """Scenario: You are a market research analyst. Your client wants to identify the single most innovative feature in a recently launched smartphone that is not available in its direct competitor's flagship model.
    
        Task:
        
        Identify the two flagship smartphones in 2025 year from two major competing brands (e.g., Apple and Samsung).
        
        For each phone, use web search tools to find three detailed technical reviews from reputable sources.
        
        From these reviews, compile a list of the top five advertised features for each phone.
        
        Compare the two lists to identify a feature present in one phone but absent in the other.
        
        Conduct a final search to determine if this unique feature is genuinely innovative (i.e., a new application of technology in the smartphone market) or an iteration of existing tech.
        
        Present the single most innovative feature and provide a one-paragraph justification for your choice, citing the sources you used."""
    )

    start_time = time.time()
    last_step = None
    last_state = None
    for step in graph.stream({"input": question},
                             config={"callbacks": [langfuse_handler]}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        last_state = step_state
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get root from the last observed state regardless of whether last step was 'start' or 'expand'
    root = last_state["root"] if last_state is not None else next(iter(last_step.values()))["root"]
    solution_node = root.get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    print(best_trajectory[-1].content)

    print(f"Method finished in: {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    main()
