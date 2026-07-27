import os
import glob
from src.state import PipelineState
from src.sandbox_executor import run_in_sandbox


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

    return {
        "execution_success": True,
        "execution_stdout": result["stdout"],
        "execution_stderr": "",
        "agent4_error": "",
        "trained_model_path": os.path.join(result["run_dir"], "model.joblib"),
        "generated_plots": plot_paths,
    }