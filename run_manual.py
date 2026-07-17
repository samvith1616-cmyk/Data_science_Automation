from src.agents.agent1_preprocessing import agent1_preprocessing
from src.agents.agent2_eda import agent2_eda
from src.agents.agent3_code_generation  import agent3_model_selector
from src.agents.package_approval import package_approval
from src.agents.agent4_execution import agent4_execute
from src.agents.agent5_report import agent5_report


def run_pipeline_manually():
    state = {
        "raw_data_path": "data/raw.csv",
        "target_column": "survived",
        "eda_notes": [],
        "agent3_retry_count": 0,
    }

    print("=== Agent 1: Preprocessing ===")
    state.update(agent1_preprocessing(state))

    print("=== Agent 2: EDA ===")
    state.update(agent2_eda(state))

    print("=== Agent 3: Model Selection ===")
    state.update(agent3_model_selector(state))

    print("=== Package Approval ===")
    state.update(package_approval(state))

    if not state.get("packages_approved"):
        print("Packages not approved, stopping.")
        return state

    print("=== Agent 4: Execute ===")
    state.update(agent4_execute(state))

    if not state.get("execution_success"):
        print("Execution failed:\n", state.get("execution_stderr"))
        return state

    print("=== Agent 5: Report ===")
    state.update(agent5_report(state))

    print("\nFinal report saved to:", state.get("final_report_path"))
    return state


if __name__ == "__main__":
    final_state = run_pipeline_manually()