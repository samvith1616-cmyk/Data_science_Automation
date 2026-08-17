import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import PipelineState
from dotenv import load_dotenv
load_dotenv()



class Agent3Output(BaseModel):
    #class Agent3Output(BaseModel):
    model_name: str = Field(description="Name of the chosen ML model")
    reasoning: str = Field(description="One or two sentences on why this model fits the data")
    code: str = Field(description="Complete, runnable Python script")
    required_packages: list[str] = Field(description="Pip packages needed")
    metric_name: str = Field(description="Name of the evaluation metric used, e.g. 'accuracy', 'RMSE', 'F1'")
    higher_is_better: bool = Field(description="True if a higher metric value is better (e.g. accuracy), False if lower is better (e.g. RMSE)")


def agent3_model_selector(state: PipelineState) -> dict:
    error_context = ""
    if state.get("agent4_error"):
        error_context = (
            f"\n\nA previous version of this code failed with this error:\n"
            f"{state['agent4_error']}\n"
            f"Analyze the root cause of this specific error and fix it - "
            f"don't just retry the same code unchanged."
        )
    preference_context = ""
    if state.get("preferred_model"):
        preference_context = (
            f"\n\nThe user was unsatisfied with the previous model's accuracy "
            f"and would prefer you try: {state['preferred_model']}. "
            f"Use this model unless it's clearly unsuitable for the data, "
            f"in which case briefly explain why and pick the closest reasonable alternative."
        )
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(Agent3Output)

    prompt = ChatPromptTemplate([
        ("system",
        """
        You are an ML Engineer Agent. Given an EDA summary, pick the best
        model for the task and write a complete, self-contained Python
        script that:
        - The dataset is already mounted inside the Docker container.
        - ALWAYS load the dataset from exactly:
            /workspace/preprocessed.csv
        - Never use Windows paths.
        - Never use os.path.abspath().
        - Never infer or construct the dataset path.
        - Use:
            DATA_PATH = "/workspace/preprocessed.csv"
        - splits into train/test
        - trains your chosen model on the target column
        - give the list of packages required to execute the code
        - prints accuracy (classification) or RMSE (regression) to stdout
        - saves the trained model to /workspace/model.joblib using joblib
        - creates a folder called /workspace/plots
        - generates useful evaluation plots
        - saves all plots as PNG files inside /workspace/plots
        If given a previous error, carefully analyze what caused it and
        fix the root cause. Do not just retry the same code.
        - evaluates the model and prints the result in EXACTLY this format on its own line:
        RESULT_METRIC:<numeric_value>
        (e.g. "RESULT_METRIC:0.8134" or "RESULT_METRIC:4.52") - no other text on that line
        - also declare metric_name and higher_is_better matching the metric your code actually computes
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
            f"{preference_context}"
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
        "metric_name": result.metric_name,
        "higher_is_better": result.higher_is_better,
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