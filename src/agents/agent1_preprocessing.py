import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.state import PipelineState
from src.sandbox_executor import run_in_sandbox

load_dotenv()

MAX_INTERNAL_RETRIES = 3


class PreprocessingCodeOutput(BaseModel):
    code: str = Field(description="Complete Python script that loads /workspace/data.csv, cleans it (missing values, encoding), and saves the result to /workspace/preprocessed.csv")
    required_packages: list[str] = Field(description="Pip packages needed, e.g. ['pandas']")
    plan_summary: str = Field(description="One or two sentences describing what cleaning steps were applied and why")


def build_data_profile(df: pd.DataFrame) -> str:
    return (
        f"Shape: {df.shape}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"Missing values:\n{df.isnull().sum().to_string()}"
    )


def generate_preprocessing_code(profile: str, target_column: str, error_context: str = "") -> PreprocessingCodeOutput:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(PreprocessingCodeOutput)

    prompt = ChatPromptTemplate([
        ("system",
        """
        You are a Data Preprocessing Code Generation Agent. Given a dataset
        profile (shape, dtypes, missing value counts) and a target column,
        write a complete Python script that:
        - loads the provided dataset in given format at /workspace/data.csv with pandas or any required library
        - drops duplicate rows
        - fills missing values appropriately (median for numeric, mode/'Unknown' for categorical)
        - one-hot encodes categorical columns EXCEPT the target column
        - saves the cleaned dataframe to /workspace/preprocessed.csv (no index)
        - only uses pandas, numpy (already installed)
        Do not print anything except errors. If given a previous error, fix
        the root cause instead of repeating the same code.
        """
        ),
        ("human", "{query}")
    ])

    chain = prompt | structured_llm
    return chain.invoke({
        "query": f"target_column = '{target_column}'\n\nDataset profile:\n{profile}{error_context}"
    })


def agent1_preprocessing(state: PipelineState) -> dict:
    df = pd.read_csv(state["raw_data_path"])
    profile = build_data_profile(df)
    target_column = state["target_column"]

    error_context = ""
    success = False
    run_dir = None
    plan_summary = ""

    for attempt in range(MAX_INTERNAL_RETRIES):
        print(f"[Agent 1] Attempt {attempt + 1}/{MAX_INTERNAL_RETRIES}...")

        code_output = generate_preprocessing_code(profile, target_column, error_context)
        plan_summary = code_output.plan_summary

        result = run_in_sandbox(
            code=code_output.code,
            data_path=state["raw_data_path"],
            packages=code_output.required_packages,
            data_path_in_container="/workspace/data.csv",
        )

        print(f"[Agent 1] Success: {result['success']}")

        if result["success"]:
            success = True
            run_dir = result["run_dir"]
            break
        else:
            print(f"[Agent 1] Error:\n{result['stderr']}")
            error_context = f"\n\nPrevious attempt failed with:\n{result['stderr']}"

    if not success:
        return {
            "preprocessing_summary": "Preprocessing code failed after multiple attempts.",
        }

    output_path = f"{run_dir}/preprocessed.csv"

    return {
        "preprocessed_data_path": output_path,
        "preprocessing_summary": plan_summary,
    }


if __name__ == "__main__":
    state = {
        "raw_data_path": "data/raw.csv",
        "target_column": "survived",
    }
    result = agent1_preprocessing(state)
    print("\nPreprocessing summary:", result["preprocessing_summary"])
    print("Preprocessed data at:", result.get("preprocessed_data_path"))