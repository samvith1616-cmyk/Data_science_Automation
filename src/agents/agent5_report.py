from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import PipelineState

REPORT_SYSTEM_PROMPT = """
You are a technical report writer. Given the full context of an automated
ML pipeline run, write a crisp, well-organized summary (use markdown
headers) covering:
- Data preprocessing steps taken
- Key EDA findings
- Model chosen and why
- Final performance metrics (from the execution output)
Keep it concise - a busy stakeholder should understand the outcome in
under a minute of reading.
"""


def agent5_report(state: PipelineState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate([
        ("system", REPORT_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    chain = prompt | llm

    query = (
        f"Preprocessing summary:\n{state.get('preprocessing_summary', '')}\n\n"
        f"EDA analysis:\n{state.get('analysis_summary', '')}\n\n"
        f"Model chosen: {state.get('selected_model', '')}\n"
        f"Reasoning: {state.get('agent3_reasoning', '')}\n\n"
        f"Execution output:\n{state.get('execution_stdout', '')}"
    )

    response = chain.invoke({"query": query})
    report_text = response.content

    plot_paths = state.get("generated_plots", [])
    image_md = "\n\n".join(f"![plot]({p})" for p in plot_paths)

    full_report = f"{report_text}\n\n## Plots\n\n{image_md}"

    report_path = "data/final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    return {
        "final_report_path": report_path,
    }