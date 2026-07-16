from typing import TypedDict

class PipelineState(TypedDict, total = False):
    raw_data_path: str
    target_column: str
    preprocessed_data_path: str
    preprocessing_summary: str
    analysis_summary: str
    suggested_task: str
    selected_model: str
    agent3_reasoning: str
    generated_code: str
    agent3_retry_count: int
    required_packages: list[str]
    trained_model_path: str
    packages_approved: bool


