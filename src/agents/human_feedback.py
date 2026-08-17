from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.state import PipelineState


def human_feedback(state: PipelineState) -> dict:
    print(f"\nModel used: {state.get('selected_model')}")
    print(f"Execution output:\n{state.get('execution_stdout')}")

    answer = input("\nAre you satisfied with this accuracy? (y/n): ").strip().lower()

    if answer == "y":
        return {"user_satisfied": True}

    preferred = input(
        "Which model would you prefer instead? (press Enter to let the agent suggest one): "
    ).strip()

    if preferred:
        return {"user_satisfied": False, "preferred_model": preferred}

    # No response - let the LLM suggest an alternative model
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    prompt = ChatPromptTemplate([
        ("system",
        "You are an ML engineer. The user was unsatisfied with the current "
        "model's accuracy and gave no preference. Suggest ONE different "
        "model (not the one currently used) that might perform better, "
        "given the EDA summary. Respond with just the model name."
        ),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "query": (
            f"Current model: {state.get('selected_model')}\n"
            f"EDA summary: {state.get('analysis_summary')}\n"
            f"Execution output: {state.get('execution_stdout')}"
        )
    })

    return {"user_satisfied": False, "preferred_model": response.content.strip()}