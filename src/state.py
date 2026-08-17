from typing import TypedDict

class PipelineState(TypedDict, total=False):
    raw_data_path: str
    target_column: str
    preprocessed_data_path: str
    preprocessing_summary: str
    analysis_summary: str
    suggested_task: str
    eda_notes: list[str]
    selected_model: str
    agent3_reasoning: str
    generated_code: str
    agent3_retry_count: int
    max_agent3_retries: int
    required_packages: list[str]
    agent4_error: str
    execution_success: bool
    execution_stdout: str
    execution_stderr: str
    generated_plots: list[str]
    trained_model_path: str
    packages_approved: bool
    final_report_path: str
    status: str
    user_satisfied: bool
    preferred_model: str
    metric_name: str
    higher_is_better: bool
    current_metric_value: float
    best_metric_value: float
    best_snapshot: dict