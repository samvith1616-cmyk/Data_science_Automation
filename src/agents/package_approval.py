from src.state import PipelineState

def package_approval(state: PipelineState) -> dict:
    print(f"List of packages required: {state['required_packages']}")
    answer = input("Do you want to install these packages? (y/n): ").strip().lower()
    approved = answer in ["y","yes"]
    return {"packages_approved": approved}


if __name__ == "__main__":
    state = {"required_packages": ["pandas", "scikit-learn", "xgboost"]}
    result = package_approval(state)
    print(result)