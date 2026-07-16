import os
import uuid
import shutil
import docker
from docker.errors import ContainerError

from src.state import PipelineState

SANDBOX_IMAGE = "ml-sandbox"


# def agent4_executor(state: PipelineState) -> dict:
#     # 1. Create a unique, isolated folder for this run
#     run_id = uuid.uuid4().hex[:8]
#     run_dir = os.path.abspath(f"data/runs/{run_id}")
#     os.makedirs(run_dir, exist_ok=True)

#     # 2. Fix the data path inside the generated code so it matches
#     #    where the file will actually live inside the container
#     code = state["generated_code"].replace(
#         state["preprocessed_data_path"], "/workspace/preprocessed.csv"
#     )

#     # 3. Write the (path-corrected) script to that folder
#     script_path = os.path.join(run_dir, "script.py")
#     with open(script_path, "w", encoding="utf-8") as f:
#         f.write(code)

#     # 4. Copy the actual dataset into the same folder
#     shutil.copy(state["preprocessed_data_path"], os.path.join(run_dir, "preprocessed.csv"))

#     # 5. Build the shell command: install packages, then run the script
#     packages = " ".join(state.get("required_packages", []))
#     shell_cmd = f"pip install -q {packages} && python script.py"

#     client = docker.from_env()

#     try:
#         # 6. Run a fresh container from the ml-sandbox image
#         output = client.containers.run(
#             SANDBOX_IMAGE,
#             command=["sh", "-c", shell_cmd],
#             volumes={run_dir: {"bind": "/workspace", "mode": "rw"}},
#             working_dir="/workspace",
#             remove=True,
#             stdout=True,
#             stderr=True,
#         )
#         return {
#             "execution_success": True,
#             "execution_stdout": output.decode("utf-8", errors="replace"),
#             "execution_stderr": "",
#             "trained_model_path": os.path.join(run_dir, "model.joblib"),
#         }

#     except ContainerError as e:
#         stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
#         return {
#             "execution_success": False,
#             "execution_stdout": "",
#             "execution_stderr": stderr,
#         }
from src.state import PipelineState
from src.sandbox_executor import run_in_sandbox


def agent4_executor(state: PipelineState) -> dict:
    result = run_in_sandbox(
        code=state["generated_code"],
        data_path=state["preprocessed_data_path"],
        packages=state.get("required_packages", []),
        data_path_in_container="/workspace/preprocessed.csv",
    )

    if result["success"]:
        return {
            "execution_success": True,
            "execution_stdout": result["stdout"],
            "execution_stderr": "",
            "trained_model_path": f"{result['run_dir']}/model.joblib",
        }
    else:
        return {
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": result["stderr"],
        }


if __name__ == "__main__":
    # Manually chain state from previous agents for a standalone test
    state = {
        "generated_code": """
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    # Paths (can be edited if needed)
    data_path = 'data/preprocessed.csv'
    target_col = 'survived'
    model_path = '/workspace/model.joblib'

    # Load data
    df = pd.read_csv(data_path)

    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing for numeric data: median imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical data: most‑frequent imputation + one‑hot encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model – RandomForest handles non‑linear relationships and, with class_weight='balanced', mitigates the modest class imbalance.
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Build the full pipeline
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf_clf)
    ])

    # Train‑test split (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Save the trained pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()""",
        "preprocessed_data_path": "data/preprocessed.csv",
        "required_packages": ["pandas", "numpy", "scikit-learn", "joblib"],
    }
    result = agent4_executor(state)
    print(result)