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
- Final performance metrics (metric name and value)
Keep it concise - a busy stakeholder should understand the outcome in
under a minute of reading.
"""


def agent5_report(state: PipelineState) -> dict:
    print("[Agent 5] Generating report...")

    snapshot = state.get("best_snapshot")
    if not snapshot:
        # Fallback: no snapshot was ever captured (shouldn't normally happen
        # if execution succeeded at least once), use live state as backup.
        snapshot = {
            "selected_model": state.get("selected_model", ""),
            "metric_name": state.get("metric_name", ""),
            "metric_value": state.get("current_metric_value"),
            "execution_stdout": state.get("execution_stdout", ""),
            "generated_plots": state.get("generated_plots", []),
            "trained_model_path": state.get("trained_model_path", ""),
        }

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    prompt = ChatPromptTemplate([
        ("system", REPORT_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    chain = prompt | llm

    query = (
        f"Preprocessing summary:\n{state.get('preprocessing_summary', '')}\n\n"
        f"EDA analysis:\n{state.get('analysis_summary', '')}\n\n"
        f"Model chosen (best-performing run): {snapshot.get('selected_model', '')}\n"
        f"Metric: {snapshot.get('metric_name', '')} = {snapshot.get('metric_value', '')}\n\n"
        f"Execution output:\n{snapshot.get('execution_stdout', '')}"
    )

    response = chain.invoke({"query": query})
    report_text = response.content

    plot_paths = snapshot.get("generated_plots", [])
    image_md = "\n\n".join(f"![plot]({p})" for p in plot_paths)

    full_report = f"{report_text}\n\n## Plots\n\n{image_md}"

    report_path = "data/final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    return {
        "final_report_path": report_path,
    }