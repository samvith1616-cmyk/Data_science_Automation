import pandas as pd
from src.state import PipelineState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def build_data_frame(df : pd.DataFrame) -> str:
    return (
        f"Shape is {df.shape}\n"
        f"Dtypes is {df.dtypes.to_string()}\n"
        f"Null values is {df.isnull().sum().to_string()}"
    )

def agent1_preprocessing(state : PipelineState) -> dict:
    df = pd.read_csv(state['raw_data_path'])
    profile = build_data_frame(df)
    llm = ChatGroq(
        model= "llama-3.1-8b-instant",
        temperature= 0
    )
    prompt = ChatPromptTemplate([
        ("system",
        """
        You are a Data Preprocessing Agent.

        Responsibilities:
        - Detect missing values.
        - Recommend encoding techniques.
        - Suggest feature scaling.
        - Never modify the data yourself.
        - Return concise explanations.
        """
        
        ),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "query": f"The dataset contains {profile}"
    })

    plan_text = response.content

    df=df.drop_duplicates()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    output_path = "data/preprocessed.csv"
    df.to_csv(output_path,index=False)
    return {
        "preprocessed_data_path": output_path,
        "preprocessing_summary": plan_text,
    }

if __name__ == "__main__":
    state = {"raw_data_path": "data/raw.csv", "target_column": "survived"}
    result = agent1_preprocessing(state)
    print(result["preprocessing_summary"])
    print("Saved to:", result["preprocessed_data_path"])