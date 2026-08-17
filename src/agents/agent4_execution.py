import os
import glob
from src.state import PipelineState
from src.sandbox_executor import run_in_sandbox
from src.sandbox_executor    import extract_result_metric


def agent4_execute(state: PipelineState) -> dict:
    print("=== RUNNING agent4_execute VERSION_CHECK_42 ===")
    result = run_in_sandbox(
        code=state["generated_code"],
        data_path=state["preprocessed_data_path"],
        packages=state.get("required_packages", []),
        data_path_in_container="/workspace/preprocessed.csv",
    )

    print(f"[Agent 4] Execution success: {result['success']}")
    print(f"[Agent 4] Generated code was:\n{state['generated_code']}")
    print(f"[Agent 4] Error:\n{result['stderr']}")

    if not result["success"]:
        return {
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": result["stderr"],
            "agent4_error": result["stderr"],
        }

    plot_paths = glob.glob(os.path.join(result["run_dir"], "plots", "*.png"))
    metric_value = extract_result_metric(result["stdout"])
    higher_is_better = state.get("higher_is_better", True)
    trained_model_path = os.path.join(result["run_dir"], "model.joblib")

    update = {
        "execution_success": True,
        "execution_stdout": result["stdout"],
        "execution_stderr": "",
        "agent4_error": "",
        "trained_model_path": trained_model_path,
        "generated_plots": plot_paths,
        "current_metric_value": metric_value,
    }

    best_value = state.get("best_metric_value")
    is_better = (
        metric_value is not None and (
            best_value is None
            or (higher_is_better and metric_value > best_value)
            or (not higher_is_better and metric_value < best_value)
        )
    )

    if is_better:
        update["best_metric_value"] = metric_value
        update["best_snapshot"] = {
            "selected_model": state.get("selected_model"),
            "metric_name": state.get("metric_name"),
            "metric_value": metric_value,
            "execution_stdout": result["stdout"],
            "generated_plots": plot_paths,
            "trained_model_path": trained_model_path,
        }

    return update