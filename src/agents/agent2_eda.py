import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.state import PipelineState
from src.sandbox_executor import run_in_sandbox
from dotenv import load_dotenv
load_dotenv()


MAX_INTERNAL_RETRIES = 3


class EDACodeOutput(BaseModel):
    code: str = Field(description="Complete Python script that loads the CSV and prints EDA findings (stats, distributions, correlations) to stdout")
    required_packages: list[str] = Field(description="Pip packages the code needs, e.g. ['pandas', 'numpy']")


class EDAAnalysis(BaseModel):
    analysis_summary: str = Field(description="Interpretation of the EDA output: classification vs regression, class balance, correlations, data quality concerns")
    suggested_task: str = Field(description="One sentence recommending what Agent 3 should focus on")


def generate_eda_code(target_column: str, columns: list[str], error_context: str = "") -> EDACodeOutput:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(EDACodeOutput)

    prompt = ChatPromptTemplate([
        ("system",
        """
        You are an EDA Code Generation Agent. Write a Python script that:
        - loads the CSV at /workspace/data.csv with pandas
        - prints shape, dtypes, missing value counts
        - prints describe() stats, target column distribution, and a correlation matrix
        - only uses pandas, numpy (already installed)
        If given a previous error, fix it instead of repeating the same code.
        """
        ),
        ("human", "{query}")
    ])

    chain = prompt | structured_llm
    return chain.invoke({
        "query": f"target_column = '{target_column}'\ncolumns = {columns}{error_context}"
    })


def agent2_eda(state: PipelineState) -> dict:
    df = pd.read_csv(state["preprocessed_data_path"])
    target_column = state["target_column"]
    columns = df.columns.tolist()

    error_context = ""
    stdout = ""
    success = False

    for attempt in range(MAX_INTERNAL_RETRIES):
        print(f"[Agent 2] Attempt {attempt + 1}/{MAX_INTERNAL_RETRIES}...")

        code_output = generate_eda_code(target_column, columns, error_context)
        print("[Agent 2] Code generated, running in sandbox...")

        result = run_in_sandbox(
            code=code_output.code,
            data_path=state["preprocessed_data_path"],
            packages=code_output.required_packages,
        )

        print(f"[Agent 2] Success: {result['success']}")

        if result["success"]:
            stdout = result["stdout"]
            success = True
            break
        else:
            print(f"[Agent 2] Error:\n{result['stderr']}")
            error_context = f"\n\nPrevious attempt failed with:\n{result['stderr']}"

    if not success:
        return {
            "analysis_summary": "EDA code generation failed after multiple attempts.",
            "eda_notes": state.get("eda_notes", []) + [
                f"EDA execution failed after retries. Last error:\n{error_context}"
            ],
        }

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(EDAAnalysis)

    prompt = ChatPromptTemplate([
        ("system", "You are a senior data scientist. Interpret this EDA output and recommend what the next agent (model selection) should focus on."),
        ("human", "{query}")
    ])
    chain = prompt | structured_llm
    analysis: EDAAnalysis = chain.invoke({"query": f"EDA script output:\n{stdout}"})

    notes = state.get("eda_notes", [])
    notes.append(analysis.analysis_summary)

    return {
        "analysis_summary": analysis.analysis_summary,
        "eda_notes": notes,
        "suggested_task": analysis.suggested_task,
        "agent3_retry_count": 0,
    }


if __name__ == "__main__":
    state = {
        "preprocessed_data_path": "data/preprocessed.csv",
        "target_column": "survived",
        "eda_notes": [],
    }
    result = agent2_eda(state)
    print("\nAnalysis summary:\n", result["analysis_summary"])
    print("\nSuggested task:\n", result.get("suggested_task"))