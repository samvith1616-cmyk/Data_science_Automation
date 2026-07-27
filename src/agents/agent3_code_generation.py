import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import PipelineState
from dotenv import load_dotenv
load_dotenv()



class Agent3Output(BaseModel):
    model_name: str = Field(description="Name of the chosen ML model")
    reasoning: str = Field(description="One or two sentences on why this model fits the data")
    code: str = Field(description="Complete, runnable Python script")
    required_packages: list[str] = Field(description="List of pip package names the code imports, beyond the Python standard library, e.g. ['pandas', 'scikit-learn']")


def agent3_model_selector(state: PipelineState) -> dict:
    error_context = ""
    if state.get("agent4_error"):
        error_context = (
            f"\n\nA previous version of this code failed with this error:\n"
            f"{state['agent4_error']}\n"
            f"Analyze the root cause of this specific error and fix it - "
            f"don't just retry the same code unchanged."
        )
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(Agent3Output)

    prompt = ChatPromptTemplate([
        ("system",
        """
        You are an ML Engineer Agent. Given an EDA summary, pick the best
        model for the task and write a complete, self-contained Python
        script that:
        - loads the CSV at the given path with pandas
        - splits into train/test
        - trains your chosen model on the target column
        - give the list of packages required to execute the code
        - prints accuracy (classification) or RMSE (regression) to stdout
        - saves the trained model to /workspace/model.joblib using joblib
        If given a previous error, carefully analyze what caused it and
        fix the root cause. Do not just retry the same code.
        """
        ),
        ("human", "{query}")
    ])

    chain = prompt | structured_llm
    result: Agent3Output = chain.invoke({
        "query": (
            f"processed_data_path = '{state['preprocessed_data_path']}'\n"
            f"target_column = '{state['target_column']}'\n\n"
            f"EDA summary:\n{state['analysis_summary']}"
            f"{error_context}"
        )
    })

    if error_context:
        print(f"[Agent 3] Retrying after error:\n{state.get('agent4_error')}")
        print(f"[Agent 3] Selected model: {result.model_name}")
        print(result.code)
        print(f"[Agent 3] Reasoning: {result.reasoning}")
    return {
        "selected_model": result.model_name,
        "agent3_reasoning": result.reasoning,
        "generated_code": result.code,
        "required_packages": result.required_packages,
        "agent3_retry_count": state.get("agent3_retry_count", 0) + 1,

    }



if __name__ == "__main__":
    state = {
        "preprocessed_data_path": "data/preprocessed.csv",
        "target_column": "survived",
        "analysis_summary": """**Task Type:** This is a classification task, as the target column "survived" is binary (0 or 1).

**Class Balance:** The target column distribution shows a moderate class imbalance, with 461 (59%) instances of "not survived" (0) and 323 (41%) instances of "survived" (1). This imbalance may affect the performance of some classification models.

**Correlations:** 
- Strong negative correlation between "survived" and "pclass" (-0.33), indicating that higher social classes are less likely to survive.
- Strong positive correlation between "survived" and "fare" (0.25), suggesting that passengers who paid higher fares are more likely to survive.
- Strong negative correlation between "pclass" and "fare" (-0.55), indicating that higher social classes tend to pay lower fares.
- Moderate negative correlation between "age" and "pclass" (-0.34), suggesting that older passengers tend to be in lower social classes.
- Moderate positive correlation between "sibsp" and "parch" (0.38), indicating that passengers with more siblings/spouses are more likely to have more parents/children on board.

**Multicollinearity Risks:** 
- The correlation between "pclass" and "fare" (-0.55) is strong, which may lead to multicollinearity issues in models that include both features.
- The correlation between "sibsp" and "parch" (0.38) is moderate, which may also contribute to multicollinearity.

**Data Quality Concerns:** 
- The "age" column has a minimum value of 0.42, which may indicate incorrect or missing data.
- The "fare" column has a maximum value of 512.33, which is significantly higher than the 75th percentile (34.11). This may indicate outliers or incorrect data.
- The "sibsp" and "parch" columns have maximum values of 8 and 6, respectively, which may indicate large families or incorrect data.""",
        "agent3_retry_count": 0,
    }
    result = agent3_model_selector(state)
    print("Model:", result["selected_model"])
    print("Reasoning:", result["agent3_reasoning"])
    print("\nCode:\n", result["generated_code"])
    print("Required packages:", result["required_packages"])